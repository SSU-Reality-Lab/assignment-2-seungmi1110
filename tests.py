import os, cv2, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import features

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def save_heatmap(array, title, filename, cmap='jet'):
    plt.imshow(array, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    os.makedirs('results', exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def save_keypoints(image, keypoints, filename):
    vis = image.copy()
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(vis, (x, y), 2, (0,255,0), -1)
    os.makedirs('results', exist_ok=True)
    cv2.imwrite(filename, vis)

# -------------------------------------------------------------------
# 0️⃣ Load Images
# -------------------------------------------------------------------
img1 = cv2.imread('resources/yosemite1.jpg')
img2 = cv2.imread('resources/yosemite2.jpg')

gray1 = cv2.cvtColor(img1.astype(np.float32)/255.0, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2.astype(np.float32)/255.0, cv2.COLOR_BGR2GRAY)

# -------------------------------------------------------------------
# 1️⃣ Feature Computation (TODO1~6)
# -------------------------------------------------------------------
HKD = features.HarrisKeypointDetector() # 해리스 코너 검출기 
SFD = features.SimpleFeatureDescriptor() # Simple Descriptor (기본 패치 기반)
MFD = features.MOPSFeatureDescriptor() # MOPS descriptor(회전/스케일 정규화 패치 기반)

# TODO1: 해리스 응닶값(a), 방향값(b)
a1, b1 = HKD.computeHarrisValues(gray1)
a2, b2 = HKD.computeHarrisValues(gray2)

# TODO3: 해리스 코너 검출
d1 = HKD.detectKeypoints(img1)
d2 = HKD.detectKeypoints(img2)

# Filter weak keypoints 약한 코너 제외 (코너 강도 0.01 이상인 점만 남기기)
d1 = [kp for kp in d1 if kp.response > 0.01] 
d2 = [kp for kp in d2 if kp.response > 0.01]

# TODO4~6: 
desc_simple_1 = SFD.describeFeatures(img1, d1) # Simple Descriptor
desc_simple_2 = SFD.describeFeatures(img2, d2)
desc_mops_1 = MFD.describeFeatures(img1, d1) # MOPS Descriptor
desc_mops_2 = MFD.describeFeatures(img2, d2)

# -------------------------------------------------------------------
# 2️⃣ Visualization (TODO1, TODO3)
# -------------------------------------------------------------------
save_heatmap(a1, "Image1 - TODO1 Harris Strength", "results/img1_TODO1_harris_strength.png")
save_heatmap(a2, "Image2 - TODO1 Harris Strength", "results/img2_TODO1_harris_strength.png")

save_keypoints(img1, d1, "results/img1_TODO3_detected_keypoints.png")
save_keypoints(img2, d2, "results/img2_TODO3_detected_keypoints.png")

print("✅ Saved TODO1 & TODO3 visualizations.")

# -------------------------------------------------------------------
# 3️⃣ Matching (TODO7 - SSD, TODO8 - Ratio)
# -------------------------------------------------------------------
matcher_ssd = features.SSDFeatureMatcher() # SSD기반 매칭 객체 생성(단순 유클리드 거리)
matcher_ratio = features.RatioFeatureMatcher() # Ratio Test 기반 매칭 객체 생성(1등/2등 후보 거리비)

# ------------------------------
# TODO7 - SSD matching
# ------------------------------
# Step 1. SSD matcher를 이용해 두 이미지의 MOPS 디스크립터 매칭을 수행하시오.
matches_ssd = matcher_ssd.matchFeatures(desc_mops_1, desc_mops_2)

# Step 2. 거리(distance)를 기준으로 정렬하여 상위 150개의 매칭만 선택하시오.
matches_ssd = sorted(matches_ssd, key=lambda x: x.distance)[:150]

# Step 3. 매칭 결과를 시각화하여 PNG로 저장하시오.
ssd_vis = cv2.drawMatches(
    img1, d1, img2, d2, matches_ssd, None,
    matchColor=(0,255,0), singlePointColor=(255,0,0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite("results/TODO7_SSD_matches.png", ssd_vis)
print("✅ TODO7 (SSD) match result saved → results/TODO7_SSD_matches.png")

# ------------------------------
# TODO8 - Ratio matching
# ------------------------------
# Step 1. Ratio matcher를 이용해 두 이미지의 MOPS 디스크립터 매칭을 수행하시오.
matches_ratio = matcher_ratio.matchFeatures(desc_mops_1, desc_mops_2)

# Step 2. distance를 기준으로 정렬하여 상위 150개의 매칭만 선택하시오.
matches_ratio = sorted(matches_ratio, key=lambda x: x.distance)[:150]

# Step 3. 매칭 결과를 시각화하여 PNG로 저장하시오.
ratio_vis = cv2.drawMatches(
    img1, d1, img2, d2, matches_ratio, None,
    matchColor=(0,255,0), singlePointColor=(255,0,0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite("results/TODO8_Ratio_matches.png", ratio_vis)
print("✅ TODO8 (Ratio) match result saved → results/TODO8_Ratio_matches.png")

print("🎯 All TODO1–8 visualizations done! Files saved in 'results/'")

''' 
[TODO8 (Ratio)가 TODO7 (SSD)보다 잘 되는 이유 — 20221673 이승미]

- TODO7(SSD): SSDFeatureMatcher.matchFeatures(desc_mops_1, desc_mops_2)
  - dist = scipy.spatial.distance.cdist(desc1, desc2, 'euclidean')
  - 각 i에 대해 np.argmin(dist[i])만 채택 → cv2.DMatch.distance = dist[i, min_j]
  - 절대거리 1등만 고르므로, 반복 패턴(창살/나뭇잎 등)에서 1등·2등이 비슷하면 오탑이 늘어남.

- TODO8(Ratio): RatioFeatureMatcher.matchFeatures(desc_mops_1, desc_mops_2)
  - dist = scipy.spatial.distance.cdist(desc1, desc2, 'euclidean')
  - 각 i에 대해 np.argsort(dist[i])로 1등·2등 후보 j1, j2 추출
    SSD1 = dist[i, j1], SSD2 = dist[i, j2]
    match.distance = (SSD1 / SSD2)  # (SSD1==0이면 0으로 처리)
  - tests.py에서 sorted(..., key=lambda x: x.distance)[:150]로
    ‘비율이 작은’ 매칭만 선택 -> 모호한 매칭(SSD1≈SSD2)을 구조적으로 배제.
  => 결과적으로 SSD보다 FP가 줄고 정밀도/ROC 성능이 향상됨.
'''

