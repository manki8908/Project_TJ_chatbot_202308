load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "/Users/mankikim/Desktop/소스코드/OROGRAPHY_PREPROCESS/dat/kma_contributed.ncl"

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
    ANLTIM = "2021060100"
  timestep = "000"

  SYY = str_get_cols(ANLTIM,0,3)
  SMM = str_get_cols(ANLTIM,4,5)
  SDD = str_get_cols(ANLTIM,6,7)
  SHH = str_get_cols(ANLTIM,8,9)

;---- Data path
  indir = "../out/"
  ; --- asia
  ;filen = "DGRIB_ECMW_025D_OROG_ASIA"
  ;filen = "DGRIB_ECMW_025D_OROG_ASIA_DFSMASK"
  ; --- dfs
  filen = "DGRIB_ECMW_025D_OROG_SUBD"
  ;filen = "DGRIB_ECMW_025D_OROG_SUBD_DFSMASK"
  infile= indir+filen


;---- Analysis area
   ; .. ASIA
   ;nx = 1440   ; ORI
   ;ny = 361
   ;nx = 73    ; DFS_MASK
   ;ny = 57

   ; .. SUB domain
   nx = 40
   ny = 60
   ;nx = 40
   ;ny = 53


;====================================================================================





;====================================================================================
; Read Data
;====================================================================================

 nln = numAsciiRow(infile)
data = new((/nln,6/),"float")

 data=readAsciiTable(infile,6,"float",0)

lat = data(:,2)
lon = data(:,3)
tmp = data(:,4)


lat2d = onedtond(lat,(/ny,nx/))
lon2d = onedtond(lon,(/ny,nx/))
tmp2d = onedtond(tmp,(/ny,nx/))

lon2d@units      = "degree_east"
lon2d@long_name  = "LONGITUDE"
lon2d@_FillValue = -99999.0

lat2d@units      = "dgree_north"
lat2d@long_name  = "LATITUDE"
lat2d@_FillValue = -99999.0


  dim = dimsizes(lat2d)
 nlat = dim(0)
 nlon = dim(1)

 ;tmp2d!0     = "lat2d"
 ;tmp2d!1     = "lon2d"
 ;tmp2d@lat2d = lat2d
 ;tmp2d@lon2d = lon2d
 ;tmp2d@_FillValue = -999.

 printVarSummary(tmp2d)
 printVarSummary(lat2d)
 printVarSummary(lon2d)
 printMinMax(tmp2d,True)

 ; .. cut to draw global data
 ;tmp2d_cut = tmp2d(61:241,361:641)
 ;lat2d_cut = lat2d(61:241,361:641)
 ;lon2d_cut = lon2d(61:241,361:641)
 tmp2d_cut = tmp2d
 lat2d_cut = lat2d
 lon2d_cut = lon2d

 printVarSummary(tmp2d_cut)
 printVarSummary(lat2d_cut)
 printVarSummary(lon2d_cut)

 tmp2d_cut!0     = "lat2d"
 tmp2d_cut!1     = "lon2d"
 tmp2d_cut@lat2d = lat2d_cut
 tmp2d_cut@lon2d = lon2d_cut
 tmp2d_cut@_FillValue = -999.


;====================================================================================




;====================================================================================
; Plot
;====================================================================================


  ;type          ="x11"  ; "x11"
  type          ="png"  ; "x11"
  type@wkWidth  = 800
  type@wkHeight = 800

  output = "../out/" + filen

  wks = gsn_open_wks(type, output)
  gsn_define_colormap(wks,"OceanLakeLandSnow")

  res = True
  res@vpWidthF  = 0.85
  res@vpHeightF = 0.90
  res@vpXF      = 0.08
  res@vpYF      = 0.950
  res@gsnDraw   = False
  res@gsnFrame  = False

;---- Draw Map
  res@mpOutlineBoundarySets         = "NoBoundaries"
  res@mpNationalLineColor           = "yellow"            ; black color
  res@mpGeophysicalLineThicknessF   = 0.0
  res@mpGeophysicalLineColor        = "yellow"
  res@mpPerimOn                     = True
  res@mpPerimLineThicknessF         = 0.0
  res@mpPerimDrawOrder              = "PreDraw"
  res@mpGridAndLimbOn               = True
  res@mpOutlineOn                   = True
  res@mpGridLineDashPattern         = 2
  res@mpGridLineDashSegLenF         = 0.10           ;; dash line density
  res@mpGridSpacingF                = 10.0
  res@mpDataBaseVersion             = "HighRes"
  ;res@mpDataSetName                 = "Earth..4"
  ;res@mpFillOn                      = True
  ;res@mpLandFillColor               = "white"  ;; Land Area Color
  ;res@mpOceanFillColor              = "white"  ;; Ocean Area Color
  res@tmXTOn = False
  res@tmYROn = False
  res@tmYLLabelFontHeightF = 0.01

;; somewhere in the model itself.
  ;res@mpProjection        = "LambertConformal"
  ;res@mpLambertParallel1F = 30
  ;res@mpLambertParallel2F = 60
  ;res@mpLambertMeridianF  = 126

;; method of zooming in on map.
  res@gsnAddCyclic        = False
  res@mpLimitMode         = "LatLon"

  ; --- for ASIA domain
  ;res@mpMinLatF           = 15
  ;res@mpMaxLatF           = 60
  ;res@mpMinLonF           = 90
  ;res@mpMaxLonF           = 160
  ; --- for SUB domain
  res@mpMinLatF           = 28
  res@mpMaxLatF           = 50
  res@mpMinLonF           = 120
  res@mpMaxLonF           = 136


  res@tfDoNDCOverlay      = False

  plot = gsn_csm_map(wks,res) ;;;; MAP



;;---- Draw tmp

;;; contour fill res

  res2 = res
  res2@cnLinesOn            = True
  res2@cnLineLabelsOn       = False
  res2@cnInfoLabelOn        = False
  res2@cnFillOn             = True


  res2@cnLevelSelectionMode = "ManualLevels"        ; set manual contour levels
  res2@cnMinLevelValF       = 0.000001              ; set min contour level
  ;res2@cnMaxLevelValF       = 50000.                 ; set max contour level
  res2@cnMaxLevelValF       = 2000.                 ; set max contour level
  res2@cnLevelSpacingF      = 100.                    ; set contour spacing



  res2@lbLabelBarOn             = True
  res2@cnConstFLabelOn          = False
  res2@lbOrientation            = "Vertical"
  res2@pmLabelBarSide           = "Right"
  res2@pmLabelBarWidthF         = 0.04
  res2@pmLabelBarHeightF        = 0.9
  res2@pmLabelBarOrthogonalPosF = 0.01
  res2@lbLabelFontHeightF       = 0.010

  ;tmp_plot = gsn_csm_contour(wks,tmp2d,res2)
  tmp_plot = gsn_csm_contour(wks,tmp2d_cut,res2)


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
  amres@txFontHeightF = 0.012
  amres@txFont        = 26
  amres@txFontColor   = "Black"
  amres@txJust        = "BottomLeft"

  ;gsn_text_ndc(wks,"ECMW DGRB ASIA(FULL 2D)",il,itop+0.003,amres)
  gsn_text_ndc(wks,"ECMW DGRB SUBD(FULL 2D)",il,itop+0.003,amres)
  ;gsn_text_ndc(wks,"ECMW DGRB ASIA(DFS_MASK 2D)",il,itop+0.003,amres)
  amres@txJust        ="BottomRight"
  ;gsn_text_ndc(wks,"ECMWF_0.25D",ir+0.09,itop,amres)
  gsn_text_ndc(wks,"ECMWF_0.25D",ir,itop,amres)

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
  gsn_text_ndc(wks,"VALID: "+valid_utc+"("+a_hour+"h)",il+0.245,ib,amres)
  ;gsn_text_ndc(wks,"TIME: "+init_utc,ir+0.09,ib,amres)
  gsn_text_ndc(wks,"TIME: "+init_utc,ir,ib,amres)

;; KST
  ;amres@txFontColor = "Black"
  ;gsn_text_ndc(wks,valid_kst+"("+a_hour+"h)",il+0.245,ib-0.018,amres)
  ;;gsn_text_ndc(wks,init_kst,ir+0.09,ib-0.018,amres)
  ;gsn_text_ndc(wks,init_kst,ir-0.09,ib-0.018,amres)


;---- Draw border line of Korea Peninsula using LDPSDABA/high_korea_map.dat
  res_poly = True
  res_poly@gsLineColor = "black"
  res_poly@gsLineThicknessF  = 2

  map_path = "../dat/"
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

  overlay(plot,tmp_plot)

  draw(plot)   ; This will draw attached polylines and map
  frame(wks)   ; Advanced frame.


;====================================================================================

end
