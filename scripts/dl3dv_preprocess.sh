# tryout - 480p test set preproc
#python src/scripts/convert_dl3dv_test.py \
#--input_dir datasets_extra/dl3dv_downloads \
#--output_dir datasets_extra/dl3dv_480p \
#--img_subdir images_8 \
#2>&1 | tee datasets_extra/20250827_convert_480p_test_try001.log

python src/scripts/convert_dl3dv_test.py \
--input_dir datasets_extra/dl3dv_downloads/1K \
--output_dir datasets_extra/dl3dv_480p \
--img_subdir images_8 \
2>&1 | tee datasets_extra/20250827_dl3dv_convert_480p_test_try002.log