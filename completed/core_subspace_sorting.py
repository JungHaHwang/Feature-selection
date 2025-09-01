import re
from collections import Counter

def count_core_subspaces(input_file, output_file):
    # 코어 서브스페이스 패턴: (숫자, 숫자, 숫자, 숫자)
    pattern = re.compile(r"\(\d+, \d+, \d+, \d+\)")
    
    subspace_list = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            matches = pattern.findall(line)
            subspace_list.extend(matches)
    
    # 등장 횟수 카운트
    counter = Counter(subspace_list)
    
    # 내림차순 정렬
    sorted_counts = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    
    # 결과 저장
    with open(output_file, "w", encoding="utf-8") as f:
        for subspace, count in sorted_counts:
            f.write(f"{subspace} : {count}\n")

# 사용 예시
input_file = "core_subspaces.txt"   # 입력 txt 파일 이름
output_file = "core_subspace_sorting.txt"  # 출력 txt 파일 이름
count_core_subspaces(input_file, output_file)
