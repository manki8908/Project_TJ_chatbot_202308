
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "kma_contributed.ncl"

begin


errMsg = NhlGetErrorObjectId()
setvalues errMsg
"errLevel" : "Fatal" ; only print Fatal messages
end setvalues



;====================================================================================
; DATA SET
;====================================================================================

;---- Set Time Environment
;    ANLTIM = getenv("ANLTIM")
;  timestep = getenv("fff")
    ANLTIM = "2023041000"
  ;timestep = "000"
  
;   ANLTIM = tostring(ANLTIM1)
; timestep = tostring(fff1)
; fname = fname1

  ;timestep = "000"
  timestep = 16

  SYY = str_get_cols(ANLTIM,0,3)
  SMM = str_get_cols(ANLTIM,4,5)
  SDD = str_get_cols(ANLTIM,6,7)
  SHH = str_get_cols(ANLTIM,8,9)

;---- Data path
  ;infile="/s4/home/nimr/mankicom/post/MODL/LDPS/N768/DABA/LDPS_STND_AGRD_FIXD"
  in_dir2 = "../dat/"
  in_dir = "/Users/mankikim/Desktop/DATA_LINK/"


;---- Analysis area

   IS = 0
   IE = 621
   JS = 0
   JE = 809


;====================================================================================





;====================================================================================
; Read Data
;====================================================================================

fname = "l015_v070_erlo_unis_20kind.2023041000.nc"
f     = addfile (in_dir + fname, "r")
print(f)

uuu2d = f->UGRD_10maboveground(timestep,:,:)
vvv2d = f->VGRD_10maboveground(timestep,:,:)
wsp2d = wind_speed(uuu2d,vvv2d)
lat2d = f->latitude     ; Needed only for projection information
lon2d = f->longitude



;lat2d := lat2d(JS:JE,IS:IE)
;lon2d := lon2d(JS:JE,IS:IE)
;tmp2d := tmp2d(JS:JE,IS:IE)

;loc1= wrf_latlon_to_ij(lat2d,lon2d,37.62796,128.6446)
;loc2= wrf_latlon_to_ij(lat2d,lon2d,37.70748,128.7487)
;print(loc1)
;print(loc2)

  dim = dimsizes(lat2d)
 nlat = dim(0)
 nlon = dim(1)

 uuu2d!0     = "lat2d"
 uuu2d!1     = "lon2d"
 uuu2d@lat2d = lat2d
 uuu2d@lon2d = lon2d
 vvv2d!0     = "lat2d"
 vvv2d!1     = "lon2d"
 vvv2d@lat2d = lat2d
 vvv2d@lon2d = lon2d
 wsp2d!0     = "lat2d"
 wsp2d!1     = "lon2d"
 wsp2d@lat2d = lat2d
 wsp2d@lon2d = lon2d

printVarSummary(uuu2d)
printVarSummary(vvv2d)
printVarSummary(wsp2d)
printVarSummary(lat2d)
printVarSummary(lon2d)

printMinMax(uuu2d,False)
printMinMax(vvv2d,False)
printMinMax(wsp2d,False)


; ---- mtn read load lat/lon

lname = "mtn_load_ll_for_plt"

lines  = asciiread (in_dir2 + lname, -1, "string")  ; -1 means read whole file
;print(lines)                               ; echo input
smtn_lat   = str_get_field(lines, 4, ",") 
smtn_lon   = str_get_field(lines, 5, ",")
mtn_lat   = stringtodouble(smtn_lat)                    ; _FillValue = 9.96921e+36
mtn_lon   = stringtodouble(smtn_lon)                    ; _FillValue = 9.96921e+36

;printVarSummary(mtn_lat)
;printVarSummary(mtn_lon)
;print(mtn_lat)
;print(mtn_lon)
;nums@_FillValue = -9999.0                  

;====================================================================================




;====================================================================================
; Plot
;====================================================================================

  type          ="png"
  type@wkWidth  = 800
  type@wkHeight = 800

  wks = gsn_open_wks(type,"../out/sub_rotate_wsp_ldps_"+tostring(timestep))
  ;gsn_define_colormap(wks,"MPL_Blues")
  gsn_define_colormap(wks,"precip2_17lev")

  res = True
  res@vpWidthF  = 0.90
  res@vpHeightF = 0.90
  res@vpXF      = 0.003
  res@vpYF      = 0.950
  res@gsnDraw   = False
  res@gsnFrame  = False

;---- Draw Map
  res@mpOutlineBoundarySets         = "NoBoundaries"


;; somewhere in the model itself.
 res@mpProjection        = "LambertConformal"
 res@mpLambertParallel1F = 30
 res@mpLambertParallel2F = 60
 res@mpLambertMeridianF  = 126

;; method of zooming in on map.
  res@mpLimitMode         = "LatLon"

  ;res@mpMinLatF  = lat2d(0,0) - 2
  ;res@mpMinLonF  = lon2d(0,0) - 2
  ;res@mpMaxLatF  = lat2d(nlat-1,nlon-1) + 2
  ;res@mpMaxLonF  = lon2d(nlat-1,nlon-1) + 2

  res@mpMinLatF  = 37.5
  res@mpMinLonF  = 127.5
  res@mpMaxLatF  = 39
  res@mpMaxLonF  = 129.5

  ;res@mpMinLatF  = 38.07
  ;res@mpMinLonF  = 128.4
  ;res@mpMaxLatF  = 38.17
  ;res@mpMaxLonF  = 128.52


  ;print(lat2d(nlat-1,nlon-1) + 2)
  ;print(lon2d(nlat-1,nlon-1) + 2)

  res@tfDoNDCOverlay      = False    ; RDAPS ONLY!!!!!!!
  res@gsnAddCyclic        = False

  ;res@gsnMaximize = True

  plot = gsn_csm_map(wks,res) ;;;; MAP



; draw poly marker
mkres = True
mkres@gsMarkerIndex = 6
mkres@gsMarkerSizeF = 150
mkres@gsMarkerColor = "red"
mkres@gsMarkerThicknessF = 5.0
mk_id = gsn_add_polymarker(wks, plot, mtn_lon(0), mtn_lat(0), mkres)


;---- Draw vector

  ;cmap = read_colormap_file("BlAqGrYeOrReVi200")

  vcres                     = res      ; plot mods desired
  vcres@vcRefLengthF            = 0.050            ; ref vec length
  vcres@vcGlyphStyle            = "CurlyVector"    ; turn on curly vectors
  vcres@vcMinDistanceF          = 0.017            ; thin out vectors
  vcres@vcVectorDrawOrder         = "PostDraw"
  ;vcres@tiMainString            = "Vectors colored by a scalar map"
  ;vcres@lbLabelStride       = 2         ; plot every other colar bar label
  ;vcres@vcLevelPalette      = cmap(6:193,:)

;;wind barb resources don't apply
;;     vcres@vcGlyphStyle               = "WindBarb"
;;     vcres@vcWindBarbLineThicknessF   = 5
;;     vcres@vcWindBarbColor            = "Gray40"

  ;---- set magnitude 
  vcres@vcRefAnnoOn               = True
  vcres@vcRefMagnitudeF           = 10 
  vcres@vcRefAnnoString1          = "10"
  vcres@vcRefAnnoSide             = "Top"
  vcres@vcRefAnnoString2On        = False
  vcres@vcRefAnnoPerimOn          = False
  vcres@vcRefAnnoOrthogonalPosF   = -0.12
  vcres@vcRefAnnoParallelPosF     = 0.999
  vcres@vcRefAnnoBackgroundColor  = "yellow"
  vcres@gsnRightString            = ""
  vcres@gsnLeftString            = ""





;---- Draw latitude and longitude text label inside plot area

  lat_txt        = ispan(10,50,10)
  lon_txt        = ispan(50,190,10)

  res@tmYRValues = lat_txt
  res@tmYLValues = lat_txt
  res@tmXTValues = lon_txt
  res@tmXBValues = lon_txt

  nyv = dimsizes(res@tmYRValues)
  nxv = dimsizes(res@tmXTValues)

  res@amOn = False

  x              = new(nxv,float)
  y              = new(nyv,float)
  xlabels        = new(nxv,string)
  ylabels        = new(nyv,string)
  latlong_txt1   = new(nxv,graphic)
  latlong_txt2   = new(nyv,graphic)

  lltxt_plot = add_map_latlongTXT(wks,plot,res)   ;; function in kma_contributed.ncl


;---- Draw wind
  res2 = res

  res2@cnLinesOn            = False
  res2@cnLineLabelsOn       = False
  res2@cnInfoLabelOn        = False
  res2@cnFillOn             = True
  res2@tiMainString = ""
  res2@tiXAxisString = ""
  res2@tiYAxisString = ""

  res2@cnLevelSelectionMode = "ManualLevels"        ; set manual contour levels
  res2@cnMinLevelValF       = 0              ; set min contour level
  res2@cnMaxLevelValF       = 20.                 ; set max contour level
  res2@cnLevelSpacingF      = 1                    ; set contour spacing


;  res2@cnLevelSelectionMode  = "ExplicitLevels"
;  res2@cnLevels              = (/0.1,5,25,50,75,100,125,150,200,250,300,350,400,\
;                                450,500,550,600,650,700,750,800,850,900,\
;                                950,1000,1100,1200,1300,1400,1500/)

  res2@lbLabelBarOn             = True
  res2@cnConstFLabelOn          = False
  res2@lbOrientation            = "Vertical"
  res2@pmLabelBarSide           = "Right"
  res2@pmLabelBarWidthF         = 0.04
  res2@pmLabelBarHeightF        = 0.9
  res2@pmLabelBarOrthogonalPosF = 0.01
  res2@lbLabelFontHeightF       = 0.010



;---- Attach the polyline
  pres                  = res          ; polyline mods desired
  pres@gsLineThicknessF = 4.0            ; line thickness   
  pres@gsLineColor      = "red"         ; color of lines

  ;n_mll = dimsizes(mtn_lat)
  ;do n = 0, dimsizes(mtn_lat)-2
  ;  dum1 = gsn_add_polyline(wks,plot, (/mtn_lon(n), mtn_lon(n+1)/) , (/mtn_lat(n), mtn_lat(n+1)/) ,pres)
  ;end do
  dum1 = gsn_add_polyline(wks,plot, (/mtn_lon/) , (/mtn_lat/) ,pres)



;---- Draw Title & Time Labels

  getvalues plot
    "vpWidthF"  : vpw
    "vpHeightF" : vph
    "vpXF"      : vpx
    "vpYF"      : vpy
  end getvalues


  ib   = 1.0-vph-(1.0-vpy)-0.003  ;; bottom y-point
  itop = vpy                      ;; top    y-point
  ir   = vpw+0.1                  ;; right  x-point
  il   = vpx

  

  run_date = ANLTIM
  utc_date = DateText(run_date,"UTC")
  kst_date = DateText(GMT2KST(run_date),"KST")
    a_hour = "+" + timestep


;;  Title
  amres  = True
  amres@txFontHeightF = 0.014
  amres@txFont        = 26
  amres@txFontColor   = "Black"
  amres@txJust        = "BottomLeft"

  gsn_text_ndc(wks,"Windvector(m/s)",il,itop,amres)
  amres@txJust        ="BottomRight"
  gsn_text_ndc(wks,"FastBarnes OA field(0.5km; Background=LDAPS 1.5km L70)",ir-0.075,itop,amres)



;;  Time Label
valid_date = DateCalcPlus(run_date,stringtointeger(timestep))  ; new func
valid_utc = DateText(valid_date,"UTC")
 init_utc = DateText(run_date,"UTC")
valid_kst = DateText(GMT2KST(valid_date),"KST")
 init_kst = DateText(GMT2KST(run_date),"KST")


amres@txFont        = 21
amres@txFontHeightF = 0.011

;; UTC
amres@txFontColor = "Red"
amres@txJust      ="TopRight"
;gsn_text_ndc(wks,"VALID: "+valid_utc+"("+a_hour+"h)",il+0.245,ib,amres)
gsn_text_ndc(wks,"VALID: "+valid_utc+"("+a_hour+"h)",il+0.245,ib,amres)

;gsn_text_ndc(wks,"TIME: "+init_utc,ir+0.09,ib,amres)
gsn_text_ndc(wks,"TIME: "+init_utc,ir-0.09,ib,amres)




;---- Draw border line of Korea Peninsula using LDPSDABA/high_korea_map.dat
  res_poly = True
  res_poly@gsLineThicknessF  = 1.5
  map_path = "./"
     fname = map_path+"high_korea_map.dat"
      nrow = numAsciiRow(fname)
    mpdata = asciiread(fname, (/nrow, 2/), "float")
        nm = 0
        ng = new(62607, graphic)

  do while ( nm .lt. nrow )

   np = floattointeger( mpdata(nm,0) )
   undef("xx")
   undef("yy")
   xx = new(np,"float")
   yy = new(np,"float")

   nstart = nm+1

   xx(0) = mpdata(nstart,0)
   yy(0) = mpdata(nstart,1)

   do i= nstart+1, np+nm
    xx(i-nstart) = mpdata(i,0)
    yy(i-nstart) = mpdata(i,1)
   end do

   ng(nm) =gsn_add_polyline(wks,plot,xx,yy,res_poly)

   nm = nm+np+1

   delete(xx)
   delete(yy)

  end do

  aki_plot1 = gsn_csm_contour(wks,wsp2d,res2)
  plot_v=gsn_csm_vector(wks,uuu2d,vvv2d,vcres)  ; create plot

  overlay(plot,aki_plot1)
  overlay(plot,plot_v)

  draw(plot)   ; This will draw attached polylines and map
  frame(wks)   ; Advanced frame.


;====================================================================================


end

