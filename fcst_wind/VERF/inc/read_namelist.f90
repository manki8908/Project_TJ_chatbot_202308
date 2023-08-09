SUBROUTINE READ_NAMELIST(Infile, tran_data_per, test_data_per, tran_num_his, test_num_his, num_fct, &
                         num_layers, hidden_size, learning_rate, num_epoch, stn_id, patience )

 IMPLICIT NONE

 CHARACTER(LEN=200), INTENT(IN) :: Infile
 CHARACTER(LEN=200), INTENT(OUT) :: tran_data_per
 CHARACTER(LEN=200), INTENT(OUT) :: test_data_per
 INTEGER, INTENT(OUT) :: tran_num_his
 INTEGER, INTENT(OUT) :: test_num_his
 INTEGER, INTENT(OUT) :: num_fct
 INTEGER, INTENT(OUT) :: num_layers
 INTEGER, INTENT(OUT) :: num_epoch
 INTEGER, INTENT(OUT) :: hidden_size
 INTEGER, INTENT(OUT) :: stn_id
 REAL, INTENT(OUT) :: learning_rate
 REAL, INTENT(OUT) :: patience
 !INTEGER, INTENT(OUT) :: num_directions

 INTEGER :: stat



 NAMELIST /data_set/tran_data_per, test_data_per, tran_num_his, test_num_his, num_fct, stn_id
 NAMELIST /hyper_para/num_layers, hidden_size, learning_rate, num_epoch, patience

 OPEN(unit=21, file=TRIM(Infile), status='old', &
      form='formatted', iostat=stat)

  IF (stat /= 0) THEN
     PRINT '(A)', 'Error opening NAMELIST file.'
     STOP 'namelist_file_open'
  ELSE 
     PRINT '(2A)', 'Read NAMELIST file: ', TRIM(Infile)
     READ(unit=21, NML=data_set)
     READ(unit=21, NML=hyper_para)
     PRINT *, 'tran_data_per: ', tran_data_per
     PRINT *, 'test_data_per: ', test_data_per
     PRINT *, 'tran_num_his: ', tran_num_his
     PRINT *, 'test_num_his: ', test_num_his
     PRINT *, 'num_fct: ', num_fct
     PRINT *, 'stn_id: ', stn_id
     PRINT *, 'num_layers: ', num_layers
     PRINT *, 'hidden_size: ', hidden_size
     PRINT *, 'learning_rate: ', learning_rate
     PRINT *, 'num_epoch: ', num_epoch
     PRINT *, 'patience: ', patience
  ENDIF

 CLOSE(unit=21)

END SUBROUTINE READ_NAMELIST
