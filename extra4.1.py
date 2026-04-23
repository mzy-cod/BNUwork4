import taichi as ti
import taichi.math as tm

# 初始化 Taichi，建议使用 GPU 加速以获得流畅的实时渲染体验
ti.init(arch=ti.gpu)

# 屏幕分辨率
res_x = 800
res_y = 800
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# --- 场景全局参数定义 ---
# 摄像机与光源参数
camera_pos = tm.vec3(0.0, 0.0, 5.0)
light_pos = tm.vec3(-3.0, 0.0, -1.0)
light_color = tm.vec3(1.0, 1.0, 1.0)
bg_color = tm.vec3(0.0, 0.5, 0.5)  # 深青色背景

# 几何体参数
# 红色球体
sphere_center = tm.vec3(-1.2, -0.2, 0.0)
sphere_radius = 1.2
sphere_color = tm.vec3(0.8, 0.1, 0.1)

# 紫色圆锥
cone_apex = tm.vec3(1.2, 1.2, 0.0)
cone_base_y = -1.4
cone_base_radius = 1.2
cone_color = tm.vec3(0.6, 0.2, 0.8)

@ti.func
def intersect_sphere(ray_o, ray_d):
    """ 计算光线与球体的交点，返回最小正数 t """
    t = 1e10
    oc = ray_o - sphere_center
    a = tm.dot(ray_d, ray_d)
    b = 2.0 * tm.dot(oc, ray_d)
    c = tm.dot(oc, oc) - sphere_radius * sphere_radius
    discriminant = b * b - 4.0 * a * c
    
    if discriminant >= 0:
        sqrt_d = ti.sqrt(discriminant)
        t1 = (-b - sqrt_d) / (2.0 * a)
        t2 = (-b + sqrt_d) / (2.0 * a)
        if t1 > 0:
            t = t1
        elif t2 > 0:
            t = t2
    return t

@ti.func
def intersect_cone(ray_o, ray_d):
    """ 计算光线与圆锥（含底面）的交点，返回最小正数 t 和交点法向量 N """
    t = 1e10
    N = tm.vec3(0.0, 0.0, 0.0)
    
    # 圆锥数学模型推导
    H = cone_apex.y - cone_base_y
    k = cone_base_radius / H
    k2 = k * k
    oc = ray_o - cone_apex
    
    # 二次方程系数 a, b, c
    a = ray_d.x**2 + ray_d.z**2 - k2 * ray_d.y**2
    b = 2.0 * (ray_d.x * oc.x + ray_d.z * oc.z - k2 * ray_d.y * oc.y)
    c = oc.x**2 + oc.z**2 - k2 * oc.y**2
    
    discriminant = b * b - 4.0 * a * c
    
    # 检查圆锥侧面交点
    if discriminant >= 0:
        sqrt_d = ti.sqrt(discriminant)
        t1 = (-b - sqrt_d) / (2.0 * a)
        t2 = (-b + sqrt_d) / (2.0 * a)
        
        # 优先检查较近的交点 t1
        if t1 > 0:
            P = ray_o + t1 * ray_d
            if cone_base_y <= P.y <= cone_apex.y:
                t = t1
                # 法向量: 梯度归一化 (偏x, 偏y, 偏z)
                N = tm.normalize(tm.vec3(2.0*(P.x - cone_apex.x), -2.0*k2*(P.y - cone_apex.y), 2.0*(P.z - cone_apex.z)))
        
        # 若 t1 不合法，检查较远的交点 t2（例如光线从内部射出或穿透侧面）
        if t == 1e10 and t2 > 0:
            P = ray_o + t2 * ray_d
            if cone_base_y <= P.y <= cone_apex.y:
                t = t2
                N = tm.normalize(tm.vec3(2.0*(P.x - cone_apex.x), -2.0*k2*(P.y - cone_apex.y), 2.0*(P.z - cone_apex.z)))
                
    # 检查圆锥底面 (y = cone_base_y 平面)
    if ti.abs(ray_d.y) > 1e-5:
        t_cap = (cone_base_y - ray_o.y) / ray_d.y
        if 0 < t_cap < t:
            P = ray_o + t_cap * ray_d
            # 判断交点是否在底面圆内
            if (P.x - cone_apex.x)**2 + (P.z - cone_apex.z)**2 <= cone_base_radius**2:
                t = t_cap
                N = tm.vec3(0.0, -1.0, 0.0) # 底面法向朝下
                
    return t, N

@ti.func
def compute_blinn_phong(P, N, V, obj_color, Ka, Kd, Ks, shininess):
    """ Blinn-Phong 着色器计算 """
    # 归一化光线方向 L
    L = tm.normalize(light_pos - P)
    
    # 【改动 1】计算半程向量 H
    H = tm.normalize(L + V)
    
    # 1. 纯环境光 (Ambient)
    ambient = Ka * light_color * obj_color
    
    # 2. 漫反射 (Diffuse) - Lambert定律 (与 Phong 保持一致)
    diff_factor = ti.max(0.0, tm.dot(N, L))
    diffuse = Kd * diff_factor * light_color * obj_color
    
    # 3. 镜面高光 (Specular) - Blinn-Phong 核心修改
    spec_factor = 0.0
    if tm.dot(N, L) > 0.0:  # 只有在迎光面才计算高光
        # 【改动 2】高光强度由 N 和 H 的夹角余弦决定
        spec_factor = ti.pow(ti.max(0.0, tm.dot(N, H)), shininess)
        
    specular = Ks * spec_factor * light_color
    
    return ambient + diffuse + specular

@ti.kernel
def render(Ka: ti.f32, Kd: ti.f32, Ks: ti.f32, shininess: ti.f32):
    # 遍历每个像素，发射射线
    for i, j in pixels:
        # 将屏幕坐标 [0, 800] 映射到 NDC 坐标 [-1, 1]
        u = (i + 0.5) / res_x * 2.0 - 1.0
        v = (j + 0.5) / res_y * 2.0 - 1.0
        
        # 计算射线方向 (假设相机视角 FOV，约60度)
        fov_scale = ti.tan(tm.radians(30.0))
        ray_d = tm.normalize(tm.vec3(u * fov_scale, v * fov_scale, -1.0))
        
        # --- 光线求交与深度测试 (Z-buffer) ---
        t_sphere = intersect_sphere(camera_pos, ray_d)
        t_cone, N_cone = intersect_cone(camera_pos, ray_d)
        
        min_t = 1e10
        hit_obj = 0  # 0: 无, 1: 球体, 2: 圆锥
        
        if t_sphere < min_t:
            min_t = t_sphere
            hit_obj = 1
        if t_cone < min_t:
            min_t = t_cone
            hit_obj = 2
            
        # --- 最终着色 ---
        pixel_color = bg_color
        if hit_obj > 0:
            P = camera_pos + min_t * ray_d  # 世界坐标系下交点
            V = tm.normalize(camera_pos - P) # 视线方向（由交点指向摄像机）
            
            if hit_obj == 1:
                # 球体法线
                N = tm.normalize(P - sphere_center)
                pixel_color = compute_blinn_phong(P, N, V, sphere_color, Ka, Kd, Ks, shininess)
            elif hit_obj == 2:
                # 圆锥法线直接使用求交时返回的结果
                N = N_cone
                pixel_color = compute_blinn_phong(P, N, V, cone_color, Ka, Kd, Ks, shininess)
                
        # 限制 RGB 在 0 到 1 之间
        pixels[i, j] = tm.clamp(pixel_color, 0.0, 1.0)

def main():
    # --- 任务 4：完成 UI 交互面板 ---
    window = ti.ui.Window("Taichi Phong Shading", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    # 初始化 UI 参数
    Ka = 0.2
    Kd = 0.7
    Ks = 0.5
    shininess = 32.0

    while window.running:
        # 调用核心渲染 Kernel
        render(Ka, Kd, Ks, shininess)
        canvas.set_image(pixels)
        
        # 渲染 UI 面板
        with gui.sub_window("Shader Parameters", 0.05, 0.05, 0.4, 0.2):
            Ka = gui.slider_float("Ka (Ambient)", Ka, 0.0, 1.0)
            Kd = gui.slider_float("Kd (Diffuse)", Kd, 0.0, 1.0)
            Ks = gui.slider_float("Ks (Specular)", Ks, 0.0, 1.0)
            shininess = gui.slider_float("Shininess", shininess, 1.0, 128.0)
            
        window.show()

if __name__ == "__main__":
    main()