#!/bin/csh

  rm read.pyf
  f2py read.f90 -m readf90 -h read.pyf
  f2py -c read.pyf read.f90

  rm read_plus_date.pyf
  f2py read_plus_date.f90 -m read_plus_datef90 -h read_plus_date.pyf
  f2py -c read_plus_date.pyf read_plus_date.f90

  rm read_namelist.pyf
  f2py read_namelist.f90 -m read_namelistf90 -h read_namelist.pyf
  f2py -c read_namelist.pyf read_namelist.f90
