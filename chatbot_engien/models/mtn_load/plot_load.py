from pyproj import Transformer
import json
import re
from re import search, findall, match, sub
import pandas as pd
import folium
import numpy as np
import pandas as pd
import openpyxl




class find_load():

    def __init__(self, geo_data):

        self.geo_data = geo_data
        self.df_geo = None

        try: 
            self.df_geo = json.load(open(self.geo_data, encoding='utf-8'))
        except:
            self.df_geo = json.load(open(self.geo_data, econding='utf-8-sig'))   # return dict        

        
    def get_latlon(self, mload_name):

        mtn_name = mload_name.split(',')[0].strip()
        load_name = mload_name.split(',')[1].strip()

        if mtn_name == '설악산': mtn_name = '설악산_대청봉'

        self.mtn_name = mtn_name
        self.load_name = load_name

        coordinate = None

        for i in range(len(self.df_geo['features'][:])):
            find_mtn = self.df_geo['features'][i]['properties']['MNTN_NM']
            find_load = self.df_geo['features'][i]['properties']['PMNTN_NM']

            if (find_mtn == self.mtn_name) & (find_load == self.load_name):
                coordinate = pd.DataFrame(self.df_geo['features'][i]['geometry']['coordinates']).melt()
                coordinate = pd.DataFrame([ coordinate.iloc[i,1] for i in range(len(coordinate)) ], dtype=np.float64, columns=['X','Y'])
            
        
        # 좌표계 변환
        try:
            tran_5179_to_4326 = Transformer.from_crs("EPSG:5179", "EPSG:4326")
            coordinate['lat']= tran_5179_to_4326.transform(coordinate['Y'],coordinate['X'])[0]
            coordinate['lon']= tran_5179_to_4326.transform(coordinate['Y'],coordinate['X'])[1]
            print(coordinate)
        except:
            print("등산로를 찾을 수 없습니다. 재검색 해주세요")


        return coordinate

    def plot_load(self,latlon_load):

        df_asos = pd.read_excel("../DABA/META_관측지점정보_ASOS.xlsx", header=1)
        #print("before: ", df_asos.shape)
        df_asos = df_asos.drop_duplicates(['지점명'], keep='first')
        #print("after: ", df_asos.shape)

        df_asos_ext = df_asos.loc[:,['지점','지점명','위도','경도']]
        df_asos_ext['관측소'] = 'ASOS'
        df_asos_ext.head()



        df_aws = pd.read_excel("../DABA/META_관측지점정보_AWS.xlsx", header=1)
        #print("before: ", df_aws.shape)
        df_aws = df_aws.drop_duplicates(['지점명'], keep='first')
        #print("after: ", df_aws.shape)

        df_aws_ext = df_aws.loc[:,['지점','지점명','위도','경도']]
        df_aws_ext['관측소'] = 'AWS'
        df_aws_ext.head()


        df_mtn_aws = pd.read_csv("../DABA/산악AWS_지점상세정보.csv", encoding='cp949')
        #print("before: ", df_mtn_aws.shape)
        df_mtn_aws = df_mtn_aws.drop_duplicates(['위도','경도'], keep='first')
        #print("after: ", df_mtn_aws.shape)

        df_mtn_aws_ext = df_mtn_aws.loc[:,['지점번호','산이름','위도','경도']]
        df_mtn_aws_ext.columns = ['지점','지점명','위도','경도']
        df_mtn_aws_ext['관측소'] = 'MTN'
        df_mtn_aws_ext.head()


        all_stn_df = pd.concat([df_asos_ext, df_aws_ext, df_mtn_aws_ext])
        print("관측소 병합 shape: ", all_stn_df.shape)
        all_stn_df = all_stn_df.set_index('지점명')
        all_stn_df.head()




        # 위경도 영역 설정
        p1_lat, p1_lon = 38.07, 128.4
        p2_lat, p2_lon = 38.17, 128.4
        p3_lat, p3_lon = 38.17, 128.52
        p4_lat, p4_lon = 38.07, 128.52
        p5_lat, p5_lon = 38.07, 128.4
        
        
        # 중심 위도, 경도 ( 설악산 고정 )
        # lat, lon = 38.10, 128.45   # 강릉
        #lat, lon = 38.1211,	128.4606  # 설악산

        # 줌 크기
        zoom_size = 12
        
        # 구글 지도 타일 설정
        tiles = "http://mt0.google.com/vt/lyrs=p&hl=ko&x={x}&y={y}&z={z}"
        # 속성 설정
        attr = "Google"
        # 지도 객체 생성
        m = folium.Map(location = [latlon_load.loc[0,['lat']].values, latlon_load.loc[0,['lon']].values],
                       zoom_start = zoom_size,
                       tiles = tiles,
                       attr = attr)
        
        for name, lat, lon, stnif in zip(all_stn_df.index, all_stn_df.위도, all_stn_df.경도, all_stn_df.관측소):
           if stnif == 'ASOS':
               color = 'red'
           elif stnif == 'AWS':
               color = 'red'
           elif stnif == 'MTN':    
               color = 'green'
           else:
               color = 'white'  # 미확인
           folium.Marker( [lat, lon], popup=name, icon=folium.Icon(color = color) ).add_to(m) # popup: 팝업창, 마우스 클릭 정보표시
        
        folium.PolyLine([latlon_load[['lat','lon']].values], color='red', ).add_to(m)
        #folium.CircleMarker([lat,lon], popup=self.load_name, radius=2000, color='orange', fill='orange', fill_opacity=50).add_to(m)
        folium.Circle( location=[latlon_load.loc[0,['lat']].values, latlon_load.loc[0,['lon']].values], 
                      popup=self.load_name, 
                      radius=6000, 
                      color='orange', 
                      fill='orange',
                      fill_opacity=0.1).add_to(m)

        

        m.save('./map_out.html')

        return m


if __name__ == "__main__":

    geo_path = '../../DATA/FRT000801/moutain_load.geojson'
    #geo_path = "/Users/mankikim/JOB/prj_mountain/DATA/FRT000801/moutain_load.geojson"
    search_class = find_load(geo_path)
    query = '설악산, 오색리구간'
    latlon_load = search_class.get_latlon(query)
    latlon_load.to_csv("../DAOU/mtn_load_ll_for_plt", header=None)
    load_map = search_class.plot_load(latlon_load)

