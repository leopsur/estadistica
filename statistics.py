import statistics

# ==========================================
# 1. Resumen de los Cinco Números
# ==========================================
def five_number_summary(nums):
    """
    Devuelve (min, Q1, median, Q3, max) usando la librería statistics.
    """
    if not nums:
        raise ValueError("La lista de números no puede estar vacía.")
    
    v_min = min(nums)
    v_max = max(nums)
    median = statistics.median(nums)
    
    # statistics.quantiles(n=4) devuelve una lista con [Q1, Q2, Q3]
    # Extraemos el primer y tercer elemento
    q_list = statistics.quantiles(nums, n=4)
    q1 = q_list[0]
    q3 = q_list[2]
    
    return v_min, q1, median, q3, v_max


# ==========================================
# 2. Comparación de Medias y Variabilidad
# ==========================================
def compare_means(nums):
    """
    Calcula las medias aritmética, armónica y geométrica, junto con 
    la desviación poblacional y el coeficiente de variación.
    """
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


# ==========================================
# 3. Tareas con la Distribución Normal
# ==========================================
def normal_tasks(data, x, p):
    """
    Ajusta una distribución normal a los datos y calcula la CDF, 
    la inversa de la CDF y el z-score.
    """
    if not data:
        raise ValueError("El conjunto de observaciones (data) no puede estar vacío.")
    
    if len(data) < 2:
        raise ValueError("Se necesitan al menos 2 observaciones para calcular la desviación estándar.")
        
    if not (0 <= p <= 1):
        raise ValueError("La probabilidad p debe estar entre 0 y 1.")

    # Crea el objeto NormalDist estimando mu y sigma desde los datos
    dist = statistics.NormalDist.from_samples(data)
    
    # P(X <= x)
    prob_cdf = dist.cdf(x)
    
    # Valor y tal que P(X <= y) = p
    inv_cdf_val = dist.inv_cdf(p)
    
    # Z-score utilizando el método integrado de NormalDist (ó (x - dist.mean) / dist.stdev)
    z_score = dist.zscore(x)
    
    return dist, prob_cdf, inv_cdf_val, z_score
