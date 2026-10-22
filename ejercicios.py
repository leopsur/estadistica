import statistics

# 1. Resumen de los Cinco Números
def five_number_summary(nums):
    if not nums:
        raise ValueError("La lista de números no puede estar vacía.")
    v_min = min(nums)
    v_max = max(nums)
    median = statistics.median(nums)
    q_list = statistics.quantiles(nums, n=4)
    q1 = q_list[0]
    q3 = q_list[2]
    return v_min, q1, median, q3, v_max

# 2. Comparación de Medias y Variabilidad
def compare_means(nums):
    if not nums:
        raise ValueError("La lista de números no puede estar vacía.")
    if any(n <= 0 for n in nums):
        raise ValueError("Todos los números de la lista deben ser estrictamente positivos (> 0).")
    mean = statistics.mean(nums)
    harmonic_mean = statistics.harmonic_mean(nums)
    geometric_mean = statistics.geometric_mean(nums)
    pstdev = statistics.pstdev(nums)
    cv = pstdev / mean if mean != 0 else 0.0
    return {
        "mean": mean,
        "harmonic_mean": harmonic_mean,
        "geometric_mean": geometric_mean,
        "pstdev": pstdev,
        "cv": cv
    }

# 3. Tareas con la Distribución Normal
def normal_tasks(data, x, p):
    if not data:
        raise ValueError("El conjunto de observaciones (data) no puede estar vacío.")
    if len(data) < 2:
        raise ValueError("Se necesitan al menos 2 observaciones para calcular la desviación estándar.")
    if not (0 <= p <= 1):
        raise ValueError("La probabilidad p debe estar entre 0 y 1.")
    dist = statistics.NormalDist.from_samples(data)
    prob_cdf = dist.cdf(x)
    inv_cdf_val = dist.inv_cdf(p)
    z_score = dist.zscore(x)
    return dist, prob_cdf, inv_cdf_val, z_score

# ==========================================
# CÓDIGO DE PRUEBA (Para ver los resultados)
# ==========================================
if __name__ == "__main__":
    # Una lista de datos de ejemplo (estaturas ficticias, notas, etc.)
    datos_prueba = [10, 12, 15, 18, 20, 22, 25, 30]
    print("=== DATOS DE PRUEBA ===")
    print(datos_prueba)

    print("\n1. RESUMEN DE 5 NÚMEROS:")
    print("Mínimo, Q1, Mediana, Q3, Máximo ->", five_number_summary(datos_prueba))

    print("\n2. COMPARACIÓN DE MEDIAS:")
    resultados = compare_means(datos_prueba)
    for clave, valor in resultados.items():
        print(f"   {clave}: {valor:.4f}")

    print("\n3. DISTRIBUCIÓN NORMAL (Evaluando x=20, p=0.85):")
    dist, cdf, inv, z = normal_tasks(datos_prueba, x=20, p=0.85)
    print(f"   Modelo ajustado: {dist}")
    print(f"   Probabilidad P(X <= 20): {cdf:.4f}")
    print(f"   Valor para el percentil 85%: {inv:.4f}")
    print(f"   Z-score de 20: {z:.4f}")
