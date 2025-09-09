## 로그 파일 이름 형식 통일
* Testing/inference:
`{YYYY-MM-DD}_run{NNN}_{test_set_and_settings}_{method}.log`
* Training:
`{YYYY-MM-DD}_run{NNN}_{test_set_and_settings}_{method}.log`
## 저장 폴더 이름 형식 통일
* 테스트
`outputs/{YYYY-MM-DD}_run{NNN}_{test_set_and_settings}_{method}`
* 트레이닝/파인튜닝
`checkpoints/{YYYY-MM-DD}_run{NNN}_{test_set_and_settings}_{method}`
