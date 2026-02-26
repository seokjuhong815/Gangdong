# 2-point VFA T1 Mapping (NIfTI)

`5°`와 `15°` flip angle 영상(동일 공간/해상도)을 이용해 T1 map을 계산합니다.

- 기본값: `TR = 10 ms`, `TE = 1.93333333 ms`
- 입력/출력: 모두 NIfTI (`.nii` / `.nii.gz`)
- 선택사항: Hd-BET 등으로 얻은 Brain Mask 적용 가능

## 설치

```bash
pip install numpy nibabel
```

## 실행 방법

### 1) Brain mask 없이

```bash
python t1_mapping_vfa.py \
  fa5deg.nii.gz \
  fa15deg.nii.gz \
  --tr-ms 10 \
  --te-ms 1.93333333 \
  --out-t1 t1_map_ms.nii.gz \
  --out-m0 m0_map.nii.gz \
  --out-e1 e1_map.nii.gz
```

### 2) Brain mask 사용 (Hd-BET 결과 등)

```bash
python t1_mapping_vfa.py \
  fa5deg.nii.gz \
  fa15deg.nii.gz \
  --brain-mask brain_mask.nii.gz \
  --tr-ms 10 \
  --te-ms 1.93333333 \
  --out-t1 t1_map_ms_masked.nii.gz
```

## 출력

- `t1_map_ms.nii.gz`: T1 map (ms)
- `m0_map.nii.gz`: M0 map
- `e1_map.nii.gz`: `E1 = exp(-TR/T1)`

## 참고

- 본 2-point VFA 모델에서 `TE`는 식에 직접 들어가지는 않지만, acquisition 메타 정보로 인자를 유지합니다.
- 입력 두 영상과 mask는 반드시 동일한 shape이어야 합니다.
