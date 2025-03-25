def calculate_evacuation_efficiency_score(age, times, mobility_score=1.0):
    # Weights based on age groups
    if age < 10:
        w1, w2, w3, w4, w5 = 0.3, 0.1, 0.2, 0.2, 0.2
    elif 10 <= age < 50:
        w1, w2, w3, w4, w5 = 0.2, 0.3, 0.2, 0.2, 0.1
    else:
        w1, w2, w3, w4, w5 = 0.3, 0.2, 0.1, 0.2, 0.2
    
    # Calculate component scores
    time_to_exit = times.get("timeToFindExit", 0)
    s_evac = 100 / (1 + max(0, time_to_exit)) if time_to_exit > 0 else 100
    
    time_to_extinguish = times.get("timeToExtinguishFire", 0)
    s_ext = 100 / (1 + max(0, time_to_extinguish)) if time_to_extinguish > 0 else 100
    
    time_to_locate = times.get("timeToFindExtinguisher", 0)
    t_loc = 100 / (1 + max(0, time_to_locate)) if time_to_locate > 0 else 100
    
    # Cognitive response efficiency (based on alarm triggering)
    time_to_alarm = times.get("timeToTriggerAlarm", 0)
    p_cog = 100 / (1 + max(0, time_to_alarm)) if time_to_alarm > 0 else 100
    
    # Physical mobility score (normalized)
    m_phys = mobility_score * 100
    
    # Calculate EES (Evacuation Efficiency Score)
    ees = w1 * s_evac + w2 * s_ext + w3 * t_loc + w4 * p_cog + w5 * m_phys
    
    return min(max(ees, 0), 100)  # Ensure score is between 0 and 100
