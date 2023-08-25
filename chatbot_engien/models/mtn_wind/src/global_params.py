variable_info = {
    "NDNSW_surface": ['NET DOWN SURFACE SW FLUX', 'W/m^2'],
    "NCPCP_surface": ['Large-Scale Precipitation (non-convective)', 'kg/m^2'],
    "SNOL_surface": ["Large-Scale Snow", 'kg/m^2'],
    "UGRD_10m": ["U-Component of Wind", 'm/s'],
    "VGRD_10m": ["V-Component of Wind", 'm/s'],
    "TMP_1_5m": ["Temperature", 'K'],
    "TMIN_1_5m": ["Minimum Temperature", 'K'],
    "TMAX_1_5m": ["Maximum Temperature", 'K'] ,
    "SPFH_1_5m": ["Specific Humidity", 'kg/kg'],
    "RH_1_5ma": ["Relative Humidity", '%'],
    "VIS_1_5m": ["Visibility", 'm'],
    "DPT_1_5m": ["Dew Point Temperature", 'K'],
    "MAXGUST_0m": ['Maximum Wind Speed'],
    "LCDC": ["Low Cloud Cover", '%'],     # _entireatmosphere_consideredasasinglelayer_
    "MCDC": ["Medium Cloud Cover", '%'],  # _entireatmosphere_consideredasasinglelayer_
    "HCDC": ["High Cloud Cover", '%'],    # _entireatmosphere_consideredasasinglelayer_
    "TCAR": ["TOTAL CLOUD AMOUNT", '%'],  # _entireatmosphere_consideredasasinglelayer_
    "PRMSL_meansealevel": ["Pressure Reduced to MSL", "Pa"],
    "TMP_surface": ["surface Temperature", "K"],
    "PRES_surface": ["Surface Pressure", "Pa"]
}

def global_params():
    global variable_info