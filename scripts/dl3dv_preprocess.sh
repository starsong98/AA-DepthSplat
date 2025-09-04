# tryout - 480p test set preproc
#python src/scripts/convert_dl3dv_test.py \
#--input_dir datasets_extra/dl3dv_downloads \
#--output_dir datasets_extra/dl3dv_480p \
#--img_subdir images_8 \
#2>&1 | tee datasets_extra/20250827_convert_480p_test_try001.log

#python src/scripts/convert_dl3dv_test.py \
#--input_dir datasets_extra/dl3dv_downloads/1K \
#--output_dir datasets_extra/dl3dv_480p \
#--img_subdir images_8 \
#2>&1 | tee datasets_extra/20250827_dl3dv_convert_480p_test_try002.log

#python src/scripts/convert_dl3dv_test.py \
#--input_dir datasets_extra/dl3dv_downloads_3 \
#--output_dir datasets_extra/dl3dv_960p \
#--img_subdir images_4 \
#2>&1 | tee datasets_extra/20250901_dl3dv_convert_960p_test_try001.log

# manually commenting out/uncommenting L215-216
#python src/scripts/convert_dl3dv_test.py \
#--input_dir datasets_extra/dl3dv_downloads_3 \
#--output_dir datasets_extra/dl3dv_960p \
#--img_subdir images_4 \
#2>&1 | tee datasets_extra/20250901_dl3dv_convert_960p_test_try002.log

# manually commenting out/uncommenting L215-216 AND L161-162
#python src/scripts/convert_dl3dv_test.py \
#--input_dir datasets_extra/dl3dv_downloads_3 \
#--output_dir datasets_extra/dl3dv_960p \
#--img_subdir images_4 \
#2>&1 | tee datasets_extra/20250901_dl3dv_convert_960p_test_try003.log

# manually commenting out/uncommenting L215-216 AND L161-162
# this will likely result in 139 out of 140 scenes only - for some reason there is 1 scene that ruins it
# 07d9f9724ca854fae07cb4c57d7ea22bf667d5decd4058f547728922f909956b
#python src/scripts/convert_dl3dv_test.py \
#--input_dir datasets_extra/dl3dv_downloads_3 \
#--output_dir datasets_extra/dl3dv_960p \
#--img_subdir images_4 \
#2>&1 | tee datasets_extra/20250901_dl3dv_convert_960p_test_try004.log

# generating index for the converted 960p dataset (incomplete)
# 137?
#python src/scripts/generate_dl3dv_index.py \
#2>&1 | tee datasets_extra/20250901_dl3dv_generate_960p_test_try004-001.log

# if the test scenes were not processed properly, neither can the training split. wonderful.
python src/scripts/convert_dl3dv_train.py \
--input_dir datasets_extra/dl3dv_downloads_3 \
--output_dir datasets_extra/dl3dv_960p \
--img_subdir images_4 \
2>&1 | tee datasets_extra/20250901_dl3dv_convert_960p_train_try001.log

