SELECT 
    p.Land,
    p.Produktion_Mio_Tonnen,
    e.Emissionen_t_pro_Tonne,
    ROUND(p.Produktion_Mio_Tonnen * e.Emissionen_t_pro_Tonne, 2) AS Gesamt_CO2
FROM top_10produzenten p
JOIN emissionen e ON p.Land = e.Land
ORDER BY Gesamt_CO2 DESC;
