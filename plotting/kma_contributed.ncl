;add vorticity function (lambert grid)

load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"

undef("GMT2KST")
function GMT2KST(gmt:string)
local jul1,jul2,jul3,org1,kst
begin
	year = stringtointeger(str_get_cols(gmt,0,3))
	month = stringtointeger(str_get_cols(gmt,4,5))
	day= stringtointeger(str_get_cols(gmt,6,7))
	hour = stringtointeger(str_get_cols(gmt,8,9))

	jul1 = greg2jul(year,month,day,hour)
	jul2 = jul1 +0.375
	jul3 = jul2greg(jul2)
	y1 = sprinti("%0.4i",jul3(0))
	m1 = sprinti("%0.2i",jul3(1))
	d1 = sprinti("%0.2i",jul3(2))
	hh = sprinti("%0.2i",jul3(3))
	kst = y1+m1+d1+hh
	return(kst)
end
undef("DateText")
function DateText(sdate:string,htype:string)
local Year,sMonth,Month,Day,shour,odate
begin
	Year = str_get_cols(sdate,0,3)
	sMonth =(/"JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"/)
	n = stringtointeger(str_get_cols(sdate,4,5))
	Month = sMonth(n-1)
	Day = str_get_cols(sdate,6,7)
	shour = str_get_cols(sdate,8,9)

	odate = shour+htype+" "+Day+" "+Month+" "+Year

	return (odate)
end

undef("DateCalc")
function DateCalc(sdate:string)
local Year,sMDay,Month,Day,shour,odate,oy,om,d
begin

	year = stringtointeger(str_get_cols(sdate,0,3))
	month = stringtointeger(str_get_cols(sdate,4,5))
	dday = stringtointeger(str_get_cols(sdate,6,7))
	shour = stringtointeger(str_get_cols(sdate,8,10))

    sMDay = (/31,28,31,30,31,30,31,31,30,31,30,31/)

	n = mod(year,4)
	m = mod (year,100)
	k = mod (year,400)
	if ( (( n .eq. 0) .and. (m .ne. 0 ) ) .or. ( k .eq. 0) ) then
	 sMDay(1) = 29
	end if

	nday = shour /24
	ohour = shour - (nday * 24)

	d = dday + nday
	oy = year
	om = month
	if ( d .gt. sMDay(month-1)) then
	   om = month +1
	   if ( om .gt. 12) then
	     oy = year +1
		 om = 1
	   end if
	   d = d - sMDay(month-1)
    end if

	odate = sprinti("%0.4i",oy) + sprinti("%0.2i",om) + sprinti("%0.2i",d) + sprinti("%0.2i",ohour)

	return (odate)
end

undef("DateCalcPlus")
function DateCalcPlus(sdate:string,phour:integer)
;nday to phour
local Year,sMDay,Month,Day,shour,odate,oy,om,d,pday
begin

	year = stringtointeger(str_get_cols(sdate,0,3))
	month = stringtointeger(str_get_cols(sdate,4,5))
	dday = stringtointeger(str_get_cols(sdate,6,7))
	shour = stringtointeger(str_get_cols(sdate,8,9))+phour

	sMDay = (/31,28,31,30,31,30,31,31,30,31,30,31/)

	n = mod(year,4)
	m = mod (year,100)
	k = mod (year,400)
	if ( (( n .eq. 0) .and. (m .ne. 0 ) ) .or. ( k .eq. 0) ) then
		sMDay(1) = 29
	end if

	d = dday+(shour/24) 

	shour = shour-shour/24*24
	oy = year
	om = month
	if ( d .gt. sMDay(month-1)) then
	   om = month +1
	   if ( om .gt. 12) then
	     oy = year +1
		 om = 1
	   end if
	   d = d - sMDay(month-1)
    end if

	odate = sprinti("%0.4i",oy) + sprinti("%0.2i",om) + sprinti("%0.2i",d) + sprinti("%0.2i",shour)

	return (odate)
end

undef("DateCalcMinus")
function DateCalcMinus(sdate:string, shour:integer)
local Year,sMDay,Month,Day,shour,odate,oy,om,d
begin

	year = stringtointeger(str_get_cols(sdate,0,3))
	month = stringtointeger(str_get_cols(sdate,4,5))
	dday = stringtointeger(str_get_cols(sdate,6,7))
  hour = stringtointeger(str_get_cols(sdate,8,9))
  
  sMDay = (/31,28,31,30,31,30,31,31,30,31,30,31/)
    
	nday = (hour+shour) /24
	ohour = (hour+shour) - (nday * 24)

	d = dday + nday
	oy = year
	om = month
  
  if( d .le. 0) then
      if (om .eq. 1) then
         oy = oy -1
         om =12
      else
          om = om -1      
      end if
      d = sMDay(om-1) + d 
  end if

  if( ohour  .lt. 0) then
    d = d - 1
    ohour = 24 + ohour
    if( d .le. 0) then
      if (om .eq. 1) then
         oy = oy -1
         om =12
      else
          om = om -1      
      end if
      d = sMDay(om-1) + d 
    end if
  end if

	odate = sprinti("%0.4i",oy) + sprinti("%0.2i",om) + sprinti("%0.2i",d) + sprinti("%0.2i",ohour)

	return (odate)
end

undef("add_map_latlongTXT")
function add_map_latlongTXT(wks,plot,res)
local res2, bres, vpx, vpy, vpw, vph, xndc, yndc, npts, n, j, nlat, \
nlon, delta, bot_lon, top_lon, lft_lat, rgt_lat, xblabels, xbvalues, \
xtlabels, xtvalues, yllabels, ylvalues, yrlabels, yrvalues, xfix, \
xlat, xlon, yfix

begin

  res2 = res
  
  getvalues plot
     "vpXF"      :  vpx
     "vpYF"      :  vpy
     "vpWidthF"  :  vpw
     "vpHeightF" :  vph
  end getvalues
  
  npts = 100000   ; Increase to get closer match for tickmarks
  xndc = fspan(vpx,vpx+vpw,npts)
  yndc = fspan(vpy-vph,vpy,npts)
  
    n    = dimsizes(yndc) 
  xfix = new(n,float)
  yfix = new(n,float)
  xlon = new(n,float)
  xlat = new(n,float)
  delta = 0.001

;---Left axis tickmarks
  if(isatt(res2,"tmYLValues")) then
    lft_lat    = get_res_value(res2,"tmYLValues",-1)
    nlat       = dimsizes(lft_lat)
    ylvalues = new(nlat,float)
    yllabels = new(nlat,string)
    xfix  = vpx + 0.0001 ; Just a smidge into the plot to make sure we don't
                         ; get missing values returned.
;
; Loop across each left latitude value that we want a tickmark for,
; and try to find the closest X,Y NDC coordinate pair along this axis.
;
    NhlNDCToData(plot,xfix,yndc,xlon,xlat)
    do j=0,dimsizes(lft_lat)-1
      NhlNDCToData(plot,xfix,yndc,xlon,xlat)
      ii = minind(fabs(xlat-lft_lat(j)))
      if(.not.any(ismissing(ii)).and.fabs(xlat(ii)-lft_lat(j)).le.delta)
        yllabels(j) = fabs(lft_lat(j)) + ""
;        ylvalues(j) = yndc(ii(0))
        ylvalues(j) = xlon(ii(0)) 
        if(lft_lat(j).lt.0) then
;          yllabels(j) = yllabels(j) + "~S~o~N~S"
	   yllabels(j) = " "+yllabels(j) + "S"
        end if
        if(lft_lat(j).gt.0) then
;          yllabels(j) = yllabels(j) + "~S~o~N~N"
           yllabels(j) = " "+ yllabels(j) + "N" 
        end if
      else
          yllabels(j) =""      
      end if
      delete(ii)
    end do

 ; draw lat long text in left side
    tx_left = new(dimsizes(yllabels),graphic)
    txres = True
    if ( isatt(res,"tmYLLabelFont")  ) then
	txres@txFont = res2@tmYLLabelFont
    else
        txres@txFont = 1
    end if
    txres@txFontColor = 1
    if ( isatt(res2, "tmYLLabelFontHeightF")) then
	txres@txFontHeightF = res2@tmYLLabelFontHeightF
    else
	txres@txFontHeightF = 0.006
    end if

    txres@txJust        = "CenterLeft"       
    tx_left = gsn_add_text(wks,plot,yllabels,ylvalues,lft_lat,txres)
    
  end if
   

  ;---get Right axis tickmarks values
  if(isatt(res2,"tmYRValues")) then
    rgt_lat    = get_res_value(res2,"tmYRValues",-1)
    nlat       = dimsizes(rgt_lat)
    yrvalues = new(nlat,float)
    yrlabels = new(nlat,string)
    xltext = new(nlat, graphic)
 
     xfix  = vpx + vpw - 0.0001 
    
    do j=0,dimsizes(rgt_lat)-1
      NhlNDCToData(plot,xfix,yndc,xlon,xlat)
      ii = minind(fabs(xlat-rgt_lat(j)))
      if(.not.any(ismissing(ii)).and.fabs(xlat(ii)-rgt_lat(j)).le.delta)
        yrlabels(j) = fabs(rgt_lat(j)) + ""
;        yrvalues(j) = yndc(ii(0))
         yrvalues(j) = xlon(ii(0))
        if(rgt_lat(j).lt.0) then
;          yrlabels(j) = yrlabels(j) + "~S~o~N~S"
          yrlabels(j) = yrlabels(j) + "S "	  
        end if
        if(rgt_lat(j).gt.0) then
;          yrlabels(j) = yrlabels(j) + "~S~o~N~N"
          yrlabels(j) = yrlabels(j) + "N "	  
        end if
      else
         yrlabels(j) = ""
      end if
      delete(ii)
    end do  
 ; draw lat text in right side
    tx_right = new(dimsizes(yrlabels),graphic)
    txres = True
    if ( isatt(res,"tmYRLabelFont")  ) then
	txres@txFont = res2@tmYRLabelFont
    else
        txres@txFont = 1
    end if
    txres@txFontColor = 1
    if ( isatt(res2, "tmYRLabelFontHeightF")) then
	txres@txFontHeightF = res2@tmYRLabelFontHeightF
    else
	txres@txFontHeightF = 0.006
    end if

    txres@txJust        = "CenterRight"       
    tx_right = gsn_add_text(wks,plot,yrlabels,yrvalues,rgt_lat,txres)
   
   end if
   
 ;---Top axis tickmarks
  if(isatt(res2,"tmXTValues")) then
    top_lon    = get_res_value(res2,"tmXTValues",-1)
    nlon       = dimsizes(top_lon)
; convert 360 to 180 E/W   
    do il = 0,nlon-1
	if( top_lon(il) .gt. 180 ) then
		top_lon(il) = top_lon(il) -360
	end if
    end do
    
    xtvalues = new(nlon,float)
    xtlabels = new(nlon,string)
    yltext = new(nlon, graphic)
    
    yfix  = vpy - 0.0001 ; Just a smidge into the plot to make sure we don't
                         ; get missing values returned.
;
; Loop across each top longitude value that we want a tickmark for,
; and try to find the closest X,Y NDC coordinate pair along this axis.
;
    do j=0,dimsizes(top_lon)-1
      NhlNDCToData(plot,xndc,yfix,xlon,xlat)
      ii = minind(fabs(xlon-top_lon(j)))
      if(.not.any(ismissing(ii)).and.fabs(xlon(ii)-top_lon(j)).le.delta)
        xtlabels(j) = fabs(top_lon(j)) + ""
;        xtvalues(j) = xndc(ii(0))
        xtvalues(j) = xlat	(ii(0))
        if(top_lon(j).lt.0) then
;          xtlabels(j) = xtlabels(j) + "~S~o~N~W"
           xtlabels(j) = xtlabels(j) + "W"
        end if
        if(top_lon(j).gt.0) then
;          xtlabels(j) = xtlabels(j) + "~S~o~N~E"
           xtlabels(j) = xtlabels(j) + "E"
        end if
      end if
      delete(ii)
    end do    
 ; draw longitude text in top side
    tx_top = new(dimsizes(xtlabels),graphic)
    txres = True
    if ( isatt(res,"tmXTLabelFont")  ) then
	txres@txFont = res2@tmXTLabelFont
    else
        txres@txFont = 1
    end if
    txres@txFontColor = 1
    if ( isatt(res2, "tmXTLabelFontHeightF")) then
	txres@txFontHeightF = res2@tmXTLabelFontHeightF
    else
	txres@txFontHeightF = 0.006
    end if

    txres@txJust        = "TopCenter"       
    tx_top = gsn_add_text(wks,plot,xtlabels,top_lon,xtvalues-0.3,txres)
    
    end if

;---Bottom axis tickmarks
  if(isatt(res2,"tmXBValues")) then
    bot_lon    = get_res_value(res2,"tmXBValues",-1)
    nlon       = dimsizes(bot_lon)
    xbvalues = new(nlon,float)
    xblabels = new(nlon,string)

    yfix  = vpy-vph + 0.0001 ; Just a smidge into the plot to make sure 
                             ; we don't get missing values returned.
;
; Loop across each bottom longitude value that we want a tickmark for,
; and try to find the closest X,Y NDC coordinate pair along this axis.
;
    do j=0,dimsizes(bot_lon)-1
      NhlNDCToData(plot,xndc,yfix,xlon,xlat)
      ii = minind(fabs(xlon-bot_lon(j)))
      if(.not.any(ismissing(ii)).and.fabs(xlon(ii)-bot_lon(j)).le.delta)
        xblabels(j) = fabs(bot_lon(j)) + ""
;        xbvalues(j) = xndc(ii(0))
         xbvalues(j) = xlat(ii(0))
        if(bot_lon(j).lt.0) then
          xblabels(j) = xblabels(j) + "~S~o~N~W"
        end if
        if(bot_lon(j).gt.0) then
          xblabels(j) = xblabels(j) + "~S~o~N~E"
        end if
      else
          xblabels(j) =""      
      end if
      delete(ii)
    end do
    
 ; draw longitude text in bottom side
    tx_bottom = new(dimsizes(xblabels),graphic)
    txres = True
    if ( isatt(res,"tmXBLabelFont")  ) then
	txres@txFont = res2@tmXBLabelFont
    else
        txres@txFont = 1
    end if
    txres@txFontColor = 1
    if ( isatt(res2, "tmXBLabelFontHeightF")) then
	txres@txFontHeightF = res2@tmXBLabelFontHeightF
    else
	txres@txFontHeightF = 0.006
    end if

    txres@txJust        = "BottomCenter"       
    tx_bottom = gsn_add_text(wks,plot,xblabels,bot_lon,xbvalues+0.3,txres)

  end if


    return(plot)
    
end
;----------------------------------------------------------------------
; This section modifies the labelbar to have triangles at each end.
;
; Note: this code is a bit messy, and we hope to replace it with
; a single resource that you can set to turn these special triangles
; on.
;----------------------------------------------------------------------

;----------------------------------------------------------------------
; This procedure modifies the labelbar to replaced the end boxes with
; triangles.
;
; Note: this code is a bit messy, and we hope to replace it with
; a single resource that you can set to turn these special triangles
; on.
;----------------------------------------------------------------------
undef("add_labelbar_triangles")
procedure add_labelbar_triangles(wks,labelbar_id,lw,lh)
local nboxes, box1_color, box2_color, orient, vpx, vpy, vph, vpw, colors, \
box1_lft, box1_rgt, box1_top, box1_bot, box2_lft, box2_rgt, box2_top, \
box2_bot, box_mid, gnres, gnres
begin
;---Get some information about the labelbar
  getvalues labelbar_id
    "vpXF"              : vpx
    "vpYF"              : vpy
    "vpWidthF"          : vpw
    "vpHeightF"         : vph
    "lbOrientation"     : orient
    "lbLabelPosition"   : lpos       ; 0=top, 1=bottom, 2=right, 3=left
    "lbFillColors"      : colors
    "lbBoxLinesOn"      : box_lines_on
    "lbBoxLineColor"    : box_line_color
    "lbBoxMinorExtentF" : mnr_ext
    "lbBoxMajorExtentF" : mjr_ext
    "lbBottomMarginF"   : lbar_bot
    "lbTopMarginF"      : lbar_top
    "lbRightMarginF"    : lbar_rgt
    "lbLeftMarginF"     : lbar_lft
    "lbMaxLabelLenF"    : lbar_llen
    "lbLabelOffsetF"    : lbar_loff
  end getvalues

  nboxes     = dimsizes(colors)
  box1_color = colors(0)
  box2_color = colors(nboxes-1)

;--- orient=0 => horizontal labelbar
  if(orient.eq.0) then  
;---Size of each box in the labelbar
    box_size = lw/nboxes
;---Get the NDC locations of the left box
    box1_lft = vpx
    box1_rgt = vpx + box_size
    box2_lft = vpx+lw - box_size
    box2_rgt = vpx+lw
    if(lpos.eq.1) then              ; 1=bottom 
      box1_top = vpy
      box1_bot = vpy-lh
      box_mid  = vpy-lh/2.          ; triangle point
    else                            ; 0=top
      box1_top = vpy-vph+lh
      box1_bot = vpy-vph
      box_mid  = vpy-vph+(lh/2.)    ; triangle point
    end if
;---Get the NDC locations of the right box
    box2_top = box1_top
    box2_bot = box1_bot
  else                              ; Vertical labelbar
;---Size of each box in the labelbar
    box_size = lh/nboxes
;---Get the NDC locations of the bottom box
    box1_bot = vpy-lh
    box1_top = vpy-lh + box_size
    box2_top = vpy
    box2_bot = vpy - box_size

    if(lpos.eq.2) then              ; 2=right
;---Get the NDC locations of the bottom box
      box_mid  = vpx + lw/2.    ; This is for the triangle
      box1_lft = vpx
      box1_rgt = vpx + lw
    else                            ; 3=left
      box_mid  = vpx + vpw - lw/2.    ; This is for the triangle
      box1_lft = vpx + vpw - lw
      box1_rgt = vpx + vpw
    end if
;---Get the NDC locations of the top box
    box2_lft = box1_lft
    box2_rgt = box1_rgt
  end if

;---Variable to hold resources for triangle colors.
  gnres = True

;---Draw the labelbar again
  draw(labelbar_id)

;---Fill-in the two end box rectangles in background color (white)
  gnres@gsFillColor = "white"
  gsn_polygon_ndc(wks,(/box1_lft,box1_rgt,box1_rgt,box1_lft,box1_lft/),\
                      (/box1_top,box1_top,box1_bot,box1_bot,box1_top/), \
                      gnres)
  gsn_polygon_ndc(wks,(/box2_lft,box2_rgt,box2_rgt,box2_lft,box2_lft/),\
                      (/box2_top,box2_top,box2_bot,box2_bot,box2_top/),\
                      gnres)

;---Draw the two triangles, both with fill and outlines.
;
; Fill the end boxes in white to mask them out. We have to be
; clever about this in order to get rid of the box outline too.
;
  gnres@gsFillColor      = "white"
  dlt = 0.01
  if(orient.eq.0) then     ; horizontal labelbar
;---Left box
    gsn_polygon_ndc(wks,(/box1_rgt,box1_lft-dlt,box1_lft-dlt,box1_rgt,box1_rgt/),\
                        (/box1_top+dlt,box1_top+dlt,box1_bot-dlt,box1_bot-dlt,box1_top/), \
                         gnres)
;---Right box
    gsn_polygon_ndc(wks,(/box2_lft,box2_rgt+dlt,box2_rgt+dlt,box2_lft,box2_lft/),\
                        (/box2_top+dlt,box2_top+dlt,box2_bot-dlt,box2_bot-dlt,box2_top+dlt/),\
                         gnres)
  else                     ; vertical labelbar
;---Bottom box
    gsn_polygon_ndc(wks,(/box1_lft-dlt,box1_lft-dlt,box1_rgt+dlt,box1_rgt+dlt,box1_lft/),\
                        (/box1_top,box1_bot-dlt,box1_bot-dlt,box1_top,box1_top/), \
                         gnres)
;---Top box
    gsn_polygon_ndc(wks,(/box2_lft-dlt,box2_lft-dlt,box2_rgt+dlt,box2_rgt+dlt,box2_lft-dlt/),\
                        (/box2_bot,box2_top+dlt,box2_top+dlt,box2_bot,box2_bot/),\
                         gnres)
  end if

;---Fill the trianges and draw the edges in black.
  gnres@gsEdgesOn   = True
  if(orient.eq.0) then     ; horizontal labelbar
;---Left triangle
    if(box_lines_on) then
      gnres@gsEdgeColor = box_line_color
    else
      gnres@gsEdgeColor = box1_color
    end if
    gnres@gsFillColor = box1_color
    gsn_polygon_ndc(wks,(/box1_rgt,box1_rgt,box1_lft,box1_rgt/), \
                        (/box1_top,box1_bot, box_mid,box1_top/),gnres)
;---Right triangle
    if(box_lines_on) then
      gnres@gsEdgeColor = box_line_color
    else
      gnres@gsEdgeColor = box2_color
    end if
    gnres@gsFillColor = box2_color
    gsn_polygon_ndc(wks,(/box2_lft,box2_lft,box2_rgt,box2_lft/), \
                        (/box2_top,box2_bot, box_mid,box2_top/),gnres)

  else                    ; vertical labelbar
;---Bottom triangle
    if(box_lines_on) then
      gnres@gsEdgeColor = box_line_color
    else
      gnres@gsEdgeColor = box1_color
    end if
    gnres@gsFillColor = box1_color
    gsn_polygon_ndc(wks,(/box1_lft,box1_rgt, box_mid,box1_lft/), \
                        (/box1_top,box1_top,box1_bot,box1_top/),gnres)

;---Top triangle
    if(box_lines_on) then
      gnres@gsEdgeColor = box_line_color
    else
      gnres@gsEdgeColor = box2_color
    end if
    gnres@gsFillColor = box2_color
    gsn_polygon_ndc(wks,(/box2_lft,box2_rgt, box_mid,box2_lft/), \
                        (/box2_bot,box2_bot,box2_top,box2_bot/),gnres)
  end if
end

;----------------------------------------------------------------------
; This procedure recreates the labelbar associated with the given
; plot so it can replace the end boxes with triangles.
;----------------------------------------------------------------------
undef("labelbar_w_tri_ends")
procedure labelbar_w_tri_ends(wks,plot)
local views, lbres, just_strs, lbar_x, lbar_y, lbar_h, lbar_w, \
  lbar_orient, lbar_side, lbar_colors, lbar_labels, lbar_frac, lbar_ext, \
  lbar_bot, lbar_top, lbar_rgt, lbar_lft, lbar_fh, lbar_aln, lbar_ljust, \
  lbar_just, lbar_mnr, lbar_mjr, lbar_loff , lbid, nboxes, class_name, \
  labebar_object
begin
;
; This code attemps to see what kind of plot we have and
; to extract the labelbar from it.
;
  class_name = NhlClassName(plot)
  if(class_name.eq."mapPlotClass") then
    if(isatt(plot,"contour")) then
      getvalues plot@contour
        "pmAnnoViews" : views
      end getvalues
    else
      if(isatt(plot,"vector")) then
        getvalues plot@vector
          "pmAnnoViews" : views
        end getvalues
      else
        print("labelbar_w_tri_ends: Error: can't determine type of plot")
        return
      end if
    end if
  else
    getvalues plot
      "pmAnnoViews" : views
   end getvalues
  end if

;---Error checking
  if(all(ismissing(views))) then
    print("labelbar_w_tri_ends: Error: Couldn't extract labelbar associated with this plot")
    return
  end if

;---Attempt to extract labelbar
  labelbar_object = new(1,graphic)
  do i=0,dimsizes(views)-1
    if(NhlClassName(views(i)).eq."labelBarClass")  then
      labelbar_obj = views(i)
    end if
  end do

;---Error checking
  if(ismissing(labelbar_obj)) then
    print("labelbar_w_tri_ends: Error: Couldn't extract labelbar associated with this plot")
    return
  end if

;---Retrieve labelbar resources associated with plot so we can regenerate it.
  getvalues labelbar_obj
    "vpXF"                 : lbar_x
    "vpYF"                 : lbar_y
    "vpHeightF"            : lbar_h
    "vpWidthF"             : lbar_w
    "lbOrientation"        : lbar_orient
    "lbLabelPosition"      : lbar_side
    "lbFillColors"         : lbar_colors
    "lbLabelStrings"       : lbar_labels
    "lbBoxFractions"       : lbar_frac
    "lbBoxMinorExtentF"    : lbar_ext
    "lbBottomMarginF"      : lbar_bot
    "lbTopMarginF"         : lbar_top
    "lbRightMarginF"       : lbar_rgt
    "lbLeftMarginF"        : lbar_lft
    "lbLabelFontHeightF"   : lbar_fh
    "lbLabelAlignment"     : lbar_aln
    "lbLabelJust"          : lbar_ljust
    "lbJustification"      : lbar_just
    "lbBoxMinorExtentF"    : lbar_mnr
    "lbBoxMajorExtentF"    : lbar_mjr
    "lbLabelOffsetF"       : lbar_loff
  end getvalues

;
; We are going to set lbMinorExtentF to 1.0 so we know exactly where the
; labelbar boxes start and end on the frame. This means that we need to
; recalculate the width or height, depending on whether we have a vertical
; or horizontal labelbar.
;

;--- lbar_orient=0 => horizontal labelbar
  if(lbar_orient.eq.0) then  
    lbar_h = lbar_h * lbar_mnr
    lbar_y = lbar_y - 0.07
  else 
    lbar_w = lbar_w * lbar_mnr
    lbar_x = lbar_x + 0.02
  end if

;---Set some labelbar resources
  lbres                      = True

; Set initial width and height.

  lbres@vpHeightF            = lbar_h
  lbres@vpWidthF             = lbar_w

  lbres@lbLabelFontHeightF   = lbar_fh

; Allow more control over labelbars.
  lbres@lbAutoManage         = False

  lbres@lbOrientation        = lbar_orient
  lbres@lbLabelPosition      = lbar_side

; No margins around labelbar.
  lbres@lbBottomMarginF      = 0.0 ; lbar_bot
  lbres@lbLeftMarginF        = 0.0 ; lbar_lft
  lbres@lbRightMarginF       = 0.0 ; lbar_rgt
  lbres@lbTopMarginF         = 0.0 ; lbar_top

; Turn various features on and off.
  lbres@lbLabelsOn           = True
  lbres@lbPerimOn            = False
  lbres@lbTitleOn            = False
  lbres@lbMonoFillPattern    = True
;
; This specifies the proportion of the space in the direction
; perpendicular to the labelbar orientation that is occupied by the
; colorbar. By default it is 0.33, or 1/3 of the width specified by
; vpWidthF in the case of a vertically oriented labelbar.
;
  lbres@lbBoxMinorExtentF    = 1.0       ; lbar_mnr
  lbres@lbBoxMajorExtentF    = lbar_mjr

  lbres@lbBoxLinesOn         = True    ; True is the default

  lbres@lbFillColors         = lbar_colors

; Which point to position labelbar about.
  lbres@lbJustification      = lbar_just
  lbres@lbLabelAlignment     = lbar_aln
  lbres@lbLabelJust          = lbar_ljust

  lbres@lbLabelAutoStride    = True        ; NO LONGER NEEDED IN V6.1.0 

  lbres@lbLabelOffsetF       = lbar_loff + 0.3

;---Regenerate the labelbar
  nboxes = dimsizes(lbar_colors)
  lbid = gsn_create_labelbar_ndc(wks,nboxes,lbar_labels,lbar_x,lbar_y,lbres)

;---Turn off the current labelbar so we can draw the one we just regenerated.
  setvalues views(0)
    "lbLabelBarOn" : False
  end setvalues

;
; Draw plot, draw new labelbar, and then draw triangles
; over the end labelbar boxes.
;
;  draw(plot)
;	print("asdfasdfasdfasdfasdfasdfasdfasdfasdf")
;  draw(lbid)
	overlay(plot,lbid)
  add_labelbar_triangles(wks,lbid,lbar_w,lbar_h)

;  frame(wks)
end

undef("lamuv2vt")
function lamuv2vt(U:float,V:float,lat:float,lon:float)
local nx, ny, vort, distx, disty, missing
begin

					missing = -999.0
       		nx = dimsizes(V(0,:))
       		ny = dimsizes(V(:,0))

       		vort = new((/ny,nx/),float)
       		distx = new((/ny,nx/),float)
       		disty = new((/ny,nx/),float)

       		distx = missing
       		distx@_FillValue = missing

       		distx(1:ny-2,1:nx-2) = gc_latlon(lat(1:ny-2,1:nx-2),lon(1:ny-2,0:nx-3),lat(1:ny-2,1:nx-2),lon(1:ny-2,2:nx-1),2,3)
       		disty(1:ny-2,1:nx-2) = gc_latlon(lat(0:ny-3,1:nx-2),lon(1:ny-2,1:nx-2),lat(2:ny-1,1:nx-2),lon(1:ny-2,1:nx-2),2,3)
       		
       				onddisty = ndtooned(disty) 

       				if(any(onddisty .eq. 0.0))then
       				ind_disty = ind(onddisty .eq. 0.0)
       					onddisty(ind_disty) = missing
       					disty = onedtond(onddisty, dimsizes(disty))
       					end if

       				onddistx = ndtooned(distx) 

       				if(any(onddistx .eq. 0.0))then
       				ind_distx = ind(onddistx .eq. 0.0)
       					onddistx(ind_distx) = missing
       					distx = onedtond(onddistx, dimsizes(distx))
       				end if

					vort = missing
					vort@_FillValue = missing

       		vort(1:ny-2,1:nx-2) = (-V(1:ny-2,0:nx-3)+V(1:ny-2,2:nx-1))/distx(1:ny-2,1:nx-2) + (U(0:ny-3,1:nx-2)-U(2:ny-1,1:nx-2))/disty(1:ny-2,1:nx-2)
       		
       		return(vort)

end

undef("lamuv2dv")
function lamuv2dv(U:float,V:float,lat:float,lon:float)
local nx, ny, divg, distx, disty, missing
begin

					missing = -999.0
       		nx = dimsizes(V(0,:))
       		ny = dimsizes(V(:,0))

       		divg = new((/ny,nx/),float)
       		distx = new((/ny,nx/),float)
       		disty = new((/ny,nx/),float)

       		distx = missing
       		distx@_FillValue = missing

       		distx(1:ny-2,1:nx-2) = gc_latlon(lat(1:ny-2,1:nx-2),lon(1:ny-2,0:nx-3),lat(1:ny-2,1:nx-2),lon(1:ny-2,2:nx-1),2,3)
       		disty(1:ny-2,1:nx-2) = gc_latlon(lat(0:ny-3,1:nx-2),lon(1:ny-2,1:nx-2),lat(2:ny-1,1:nx-2),lon(1:ny-2,1:nx-2),2,3)
       				onddisty = ndtooned(disty) 

       				if(any(onddisty .eq. 0.0))then
       				ind_disty = ind(onddisty .eq. 0.0)
       					onddisty(ind_disty) = missing
       					disty = onedtond(onddisty, dimsizes(disty))
       					end if

       				onddistx = ndtooned(distx) 

       				if(any(onddistx .eq. 0.0))then
       				ind_distx = ind(onddistx .eq. 0.0)
       					onddistx(ind_distx) = missing
       					distx = onedtond(onddistx, dimsizes(distx))
       				end if

       		
					divg = missing
					divg@_FillValue = missing

       		divg(1:ny-2,1:nx-2) = (-U(1:ny-2,0:nx-3)+U(1:ny-2,2:nx-1))/distx(1:ny-2,1:nx-2) + (-V(0:ny-3,1:nx-2)+V(2:ny-1,1:nx-2))/disty(1:ny-2,1:nx-2)
       		
       		return(divg)

end
undef("lamuv2cv")
function lamuv2cv(U:float,V:float,lat:float,lon:float)
local missing, nx, ny, conv
begin
					missing = -999.0
       		nx = dimsizes(V(0,:))
       		ny = dimsizes(V(:,0))
       		
       		conv = lamuv2dv(U,V,lat,lon)*-1.0
       		return(conv)
end

undef("getgrb")
function getgrb(dumps, varlist:string, run_date)
local missing, var
begin
    list1 = getfilevarnames(dumps)
;    print(varlist)
        undef("var")
        var = dumps->$varlist$
;        print(var!0)
;    if(any(.not.ismissing(str_match(list1, "initial_time0"))))then
    if(var!0 .eq. "initial_time0_hours")then
        undef("var")
        init = dumps->initial_time0

        yyyy = str_get_cols(run_date, 0,3)
        mm = str_get_cols(run_date, 4,5)
        dd = str_get_cols(run_date, 6,7)
        hh = str_get_cols(run_date, 8,9)

        date_idx = mm + "/" + dd + "/" + yyyy + " (" + hh + ":00)"
 
        init = str_squeeze(init)
        date_idx = str_squeeze(date_idx)
;        print(date_idx)
;        print(init)
        init_ind = ind(init .eq. date_idx)
        ;num of dimension
;        print(list1)
        n_dims = dimsizes(dimsizes(dumps->$varlist$))
;        print(n_dims)
        if(n_dims .eq. 4)then
            var = dumps->$varlist$(init_ind , :, :, :)
        end if
        if(n_dims .eq. 3)then
            var = dumps->$varlist$(init_ind , :, :)
        end if
        
;        print(n_dims)
    else
        var = dumps->$varlist$
    end if

return(var)
end
undef("attach_labelbar")
function attach_labelbar(wks,map,arr[*]:numeric,colors[*])
local lbres, vph, vpw, nboxes, sf, ii, eqind, conarr, lbstring
begin
  getvalues map
    "vpHeightF" : vph
    "vpWidthF"  : vpw
  end getvalues

  conarr = tofloat(tointeger(arr))
  eqind  = new(dimsizes(arr),integer)

	do ii = 0, dimsizes(eqind)-1
		if((conarr(ii) .eq. arr(ii)))then
			eqind(ii) = 1
		else
			eqind(ii) = 0
		end if
	end do

	if(all(eqind .eq. 1))then
		lbstring = sprintf("%0.0f",arr)
	else
		lbstring = sprintf("%0.1f",arr)
	end if
  nboxes = dimsizes(colors)-2

  sf = ((nboxes+1)*1.0)/((nboxes)*1.0)
  lbres = True          ; labelbar only resources
  lbres@vpWidthF = 0.03     ; labelbar width
  lbres@vpHeightF = vph*sf     ; labelbar height
  lbres@lbAutoManage = False          ; Necessary to control sizes
  lbres@lbFillColors = colors(1:dimsizes(colors)-2)
  lbres@lbMonoFillPattern = True          ; Solid fill pattern
  lbres@lbLabelFontHeightF = 0.009          ; font height. default is small
  lbres@lbOrientation = "Vertical"
  lbres@lbPerimOn = False
  lbres@lbLabelAlignment = "ExternalEdges"
  lbres@lbLabelJust = "CenterRight"
  lbres@lbLabelStrings = lbstring
  lbid = gsn_create_labelbar(wks,nboxes,arr,lbres)

  amres = True
  amres@amZone = 2

  amres@amJust = "CenterLeft"
  amres@amParallelPosF =  1.005   ; Center
  amres@amOrthogonalPosF =  -1.0+(1.0/nboxes)*-0.5
  annoid = gsn_add_annotation(map,lbid,amres)
  return(annoid)
end


undef("getFillMode")
function getFillMode(plot:graphic, idata:float, cnlevel:float, cncolor:integer, res:logical)
local lat, lon, idmax, idmin, ndclat, ndclon, xx, yy, yyxx
begin
	lat = idata@lat2d
	lon = idata@lon2d
	yyxx = dimsizes(idata)

	ndclat = ndtooned(lat)
	ndclon = ndtooned(lon)
	datatondc(plot, ndtooned(lon), ndtooned(lat), ndclat, ndclon)

	xx = ind(.not. ismissing(ndclat))

	onedidata = ndtooned(idata)
	idmax = max(onedidata(xx))
	idmin = min(onedidata(xx))

        if(ismissing(idmax) .or. ismissing(idmin))then
                return(res)
        end if

	maxindx = GetFillColorIndex(cnlevel, cncolor, idmax)
	minindx = GetFillColorIndex(cnlevel, cncolor, idmin)

;	print(maxindx+", "+minindx)
;	print(idmax+", "+idmin)
	
	;print(maxindx+", "+minindx)
	;print(idmax+", "+idmin)

	if(maxindx .eq. minindx)then
		res@cnFillMode = "RasterFill"
	else
		res@cnFillMode = "AreaFill"
	end if

	return(res)

end

;;interpolation 2d data for kwn by mshan 
undef("InTer")
function InTer(idata:float, ilat:float, ilon:float,\
 olat:float, olon:float)
local missing, nx, ny, nxny, ix, iy, dist, distxy, diff_min, \
igori, jgori, onedilat, onedilon, onedidata, \
odata, udata

begin
	missing = -999.0

	;rscan = (/avg(ilon(:,:dimsizes(ilon(0,:))-2) - ilon(:,1:))*0.5 + avg(ilat(:dimsizes(ilat(:,0))-2,:) - ilat(1:,:))*0.5/)
	;rscan = abs(rscan)*2
	rscan = (/1.0/)
	odata = obj_anal_ic(ndtooned(ilon),ndtooned(ilat),ndtooned(idata),olon,olat,rscan,False)
	return(odata)
end

; Calculate meteorological wind direction
; *****************************************************************
undef ("wind_direction")
function wind_direction(u:numeric, v:numeric, opt:integer)
; meteorological wind direction.
; This is over-kill. Only need: wdir=atan2(u,v)*radi + 180
local wdir, radi, con, wcrit, wmsg, zero, dimu
begin
  if (.not.(typeof(u).eq."double" .or. \
            typeof(u).eq."float"  .or. \
            typeof(u).eq."integer" )) then
      print("wind_direction: illegal numeric type: "+typeof(u))
      exit
  end if

  dimu = dimsizes(u)

  if (typeof(u).eq."double") then
      zero = 0.0d
      wmsg = 1d20
      con  = 180.0d
      wcrit= 360d0-0.00002d0
      radi = 1.0d0/0.0174532925199433d0
      wdir = new (dimu, typeof(u), wmsg)
  else
      zero = 0.0
      wmsg = 1e20
      con  = 180.0
      wcrit= 360-0.00002
      radi = 1.0/0.01745329
      wdir = new (dimu, "float", wmsg)
  end if

  wdir = (/ atan2(u,v)*radi + con  /)
  wdir = where(wdir.ge.wcrit, zero, wdir)  ; force 360 "north winds" to 0

  if (typeof(opt).eq."integer" .and. opt.eq.0) then
      cdir = zero                          ; calm 0.0->_FillValue
      wdir = where (u.eq.zero .and. v.eq.zero, cdir, wdir)
  end if

  if (typeof(opt).eq."integer" .and. opt.eq.1) then
      cdir = wdir@_FillValue               ; calm 0.0->_FillValue
      wdir = where (u.eq.zero .and. v.eq.zero, cdir, wdir)
  end if

  if (typeof(opt).eq."float" .or. typeof(opt).eq."double") then
                                           ; calm 0.0->user_value
      wdir = where (u.eq.zero .and. v.eq.zero, opt, wdir)
  end if
  wdir@long_name = "wind direction (meteorological)"
  if (isatt(wdir,"units")) then
      delete(wdir@units)
  end if
  return(wdir)
end
  
undef ("rvf")
function rvf(u:numeric, v:numeric)
local pi, omeg, Re, lat, lon, i, j, dv, dx, du, dy, relvort
begin
	pi = 3.141592
	omeg=2*pi/(60*60*23.93447);     ; Earth rotation rate, s^-1
	Re=6.37e6;                      ; Radius of Earth, m
	lat = u&$u!0$
	lon = u&$u!1$
	relvort = new(dimsizes(u),float)

	do i = 1, dimsizes(lon)-2
	    do j=1, dimsizes(lat)-2
		dv=v(j,i+1)-v(j,i-1);
		dx=(lon(i+1)-lon(i-1))*cos(lat(j)*pi/180)*pi*Re/180;
		du=u(j-1,i)-u(j+1,i);
		dy=(lat(j+1)-lat(j-1))*pi*Re/180;
		relvort(j,i)=(dv/dx)-(du/dy);  ; in s^-1
	    end do
	end do
	return(relvort)

end

