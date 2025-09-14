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
#python src/scripts/convert_dl3dv_train.py \
#--input_dir datasets_extra/dl3dv_downloads_3 \
#--output_dir datasets_extra/dl3dv_960p \
#--img_subdir images_4 \
#2>&1 | tee datasets_extra/20250901_dl3dv_convert_960p_train_try001.log

# 480p, test split
# failed because that one scene was 135x240 instead of 270x480
#python src/scripts/convert_dl3dv_test.py \
#--input_dir datasets_extra/dl3dv_downloads_4 \
#--output_dir datasets_extra/dl3dv_480p \
#--img_subdir images_8 \
#2>&1 | tee datasets_extra/20250904_dl3dv_convert_480p_test_try001.log


### AFTER MANUAL REARRANGING

# 480p, test split
# manually switched that one scene now
# NOW it worked
#python src/scripts/convert_dl3dv_test.py \
#--input_dir datasets_extra/dl3dv_downloads_4 \
#--output_dir datasets_extra/dl3dv_480p \
#--img_subdir images_8 \
#2>&1 | tee datasets_extra/20250904_dl3dv_convert_480p_test_try002.log

# now generating index for that split
#python src/scripts/generate_dl3dv_index_480p_test.py \
#2>&1 | tee datasets_extra/20250904_dl3dv_index_480p_test_try002-001.log

# 480p, training split
#python src/scripts/convert_dl3dv_train.py \
#--input_dir datasets_extra/dl3dv_downloads_4 \
#--output_dir datasets_extra/dl3dv_480p \
#--img_subdir images_8 \
#2>&1 | tee datasets_extra/20250904_dl3dv_convert_480p_train_try001.log

# 480p, training split
# trying out new script
#python src/scripts/convert_dl3dv_train_v2.py \
#--input_dir datasets_extra/dl3dv_downloads \
#--output_dir datasets_extra/dl3dv_480p \
#--img_subdir images_8 \
#--test_scenes_json datasets_extra/dl3dv_480p/test/index.json \
#2>&1 | tee datasets_extra/20250904_dl3dv_convert_480p_train_try002.log

# 480p, training split index generation
# trying out new script
# a bunch of samples were left out, but I guess that's fair for now.
#python src/scripts/generate_dl3dv_index_v3.py \
#--dataset_path datasets_extra/dl3dv_480p \
#--stage train \
#2>&1 | tee datasets_extra/20250905_dl3dv_index_480p_train_try002-001.log


### 960P, AFTER MANUAL REARRANGING

# 960p, test split
# manually switched that one scene now
# worked well
#python src/scripts/convert_dl3dv_test_v2.py \
#--input_dir datasets_extra/dl3dv_downloads_3 \
#--output_dir datasets_extra/dl3dv_960p \
#--img_subdir images_4 \
#2>&1 | tee datasets_extra/20250910_dl3dv_convert_960p_test_try002.log

# 480p, test split index generation
# trying out (the same) new script
# all accounted for?
python src/scripts/generate_dl3dv_index_v3.py \
--dataset_path datasets_extra/dl3dv_960p \
--stage test \
2>&1 | tee datasets_extra/20250910_dl3dv_index_960p_test_try002-001.log