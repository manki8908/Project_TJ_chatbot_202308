SUBROUTINE READ_BINARY(infile, NV, NS, NH, NF, buf)

  IMPLICIT NONE

  !.. Argument
  CHARACTER(LEN=*), INTENT(IN) :: infile
  INTEGER, INTENT(IN) :: NV, NS, NH, NF
  REAL, DIMENSION(NV,NS,NH,NF), INTENT(OUT) :: buf

  !.. Local
  INTEGER :: NV1, NS1, NH1, NF1
  INTEGER, DIMENSION(:), ALLOCATABLE :: stn_id
  INTEGER, DIMENSION(:,:), ALLOCATABLE :: ndate
  INTEGER :: stat
  INTEGER, DIMENSION(4) :: sdate, edate
  INTEGER :: idate
  LOGICAL :: is_it_there
  INTEGER :: get_unit



!  IF (stat /= 0) THEN
!     PRINT *, " Can not found ", TRIM(infile)
!     RETURN
!  END IF

  get_unit = 99

  INQUIRE ( file = TRIM(infile) , exist = is_it_there )

  IF (is_it_there) THEN
     OPEN(get_unit, FILE=TRIM(infile), STATUS='OLD', FORM="UNFORMATTED", IOSTAT=stat)
  ELSE
     PRINT *, " Fortran Error: Can not found ", TRIM(infile)
     RETURN
  END IF      


  PRINT *, "========== Start Read DFS bianary", TRIM(infile)

  READ(get_unit) sdate, edate, idate
  READ(get_unit) NV1, NS1, NH1, NF1
     PRINT *, "FILE start date : ", sdate
     PRINT *, "FILE   end date : ", edate
     PRINT *, "DATE interval : ", idate
     PRINT *, "File dimension: "
     PRINT *, "FILE DIMENSION(NV) = ", NV1
     PRINT *, "FILE DIMENSION(NS) = ", NS1
     PRINT *, "FILE DIMENSION(NH) = ", NH1
     PRINT *, "FILE DIMENSION(NF) = ", NF1
     PRINT *, "User dimension: "
     PRINT *, "FILE DIMENSION(NV) = ", NV
     PRINT *, "FILE DIMENSION(NS) = ", NS
     PRINT *, "FILE DIMENSION(NH) = ", NH
     PRINT *, "FILE DIMENSION(NF) = ", NF


  IF ( NS /= NS1 .OR. NH /= NH1 .OR. NF /= NF1 ) THEN
     PRINT *, "Invaild NV,NS,NH,NF in READ_TRN_FOR_LSTM"
     RETURN
  END IF 

  IF ( ALLOCATED(stn_id) ) DEALLOCATE(stn_id)
  IF ( ALLOCATED(ndate) )  DEALLOCATE(ndate)
  ALLOCATE(stn_id(NS1))
  ALLOCATE(ndate(NH1,4))

  READ(get_unit) stn_id
!     PRINT *, "Station order : ", stn_id
  READ(get_unit) ndate
!     PRINT *, "Date order : ", ndate
  READ(get_unit) buf
!     PRINT *, "last day print"
!     PRINT '(99f7.2)', buf(1,38, NH1,:)


  CLOSE(get_unit)

  RETURN


END SUBROUTINE READ_BINARY
