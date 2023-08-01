variable_info = {
    "NDNSW_surface": ['NET DOWN SURFACE SW FLUX', 'W/m^2'],
    "NCPCP_surface": ['Large-Scale Precipitation (non-convective)', 'kg/m^2'],
    "SNOL_surface": ["Large-Scale Snow", 'kg/m^2'],
    "UGRD_10maboveground": ["U-Component of Wind", 'm/s'],
    "VGRD_10maboveground": ["V-Component of Wind", 'm/s'],
    "TMP_1_5maboveground": ["Temperature", 'K'],
    "TMIN_1_5maboveground": ["Minimum Temperature", 'K'],
    "TMAX_1_5maboveground": ["Maximum Temperature", 'K'] ,
    "SPFH_1_5maboveground": ["Specific Humidity", 'kg/kg'],
    "RH_1_5maboveground": ["Relative Humidity", '%'],
    "VIS_1_5maboveground": ["Visibility", 'm'],
    "DPT_1_5maboveground": ["Dew Point Temperature", 'K'],
    "MAXGUST_0maboveground": ['Maximum Wind Speed'],
    "LCDC_entireatmosphere_consideredasasinglelayer_": ["Low Cloud Cover", '%'],
    "MCDC_entireatmosphere_consideredasasinglelayer_": ["Medium Cloud Cover", '%'],
    "HCDC_entireatmosphere_consideredasasinglelayer_": ["High Cloud Cover", '%'],
    "TCAR_entireatmosphere_consideredasasinglelayer_": ["TOTAL CLOUD AMOUNT", '%'],
    "PRMSL_meansealevel": ["Pressure Reduced to MSL", "Pa"],
    "TMP_surface": ["surface Temperature", "K"],
    "PRES_surface": ["Surface Pressure", "Pa"]
}

def global_params():
    global variable_info