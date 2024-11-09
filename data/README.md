
# diner_df
- diner_df_20241107_071929_yamyam.csv
- 수집된 음식점 수: 37,408 개
  
| 컬럼명 | 설명 |
|---------|------|
| diner_idx | 음식점 고유 식별자 |
| diner_name | 음식점 이름 |
| diner_category_large | 대분류 카테고리 |
| diner_category_middle | 중분류 카테고리 |
| diner_category_small | 소분류 카테고리 |
| diner_category_detail | 상세 카테고리 |
| diner_tag | 음식점 태그 |
| diner_menu | 메뉴 정보 |
| diner_menu_name | 메뉴 이름 |
| diner_menu_price | 메뉴 가격 |
| diner_review_cnt | 리뷰 수 |
| diner_blog_review_cnt | 블로그 리뷰 수 |
| diner_review_avg | 평균 평점 |
| diner_review_tags | 리뷰 태그 |
| diner_address | 주소 |
| diner_phone | 전화번호 |
| diner_lat | 위도 |
| diner_lon | 경도 |
| diner_url | 음식점 URL |
| diner_open_time | 영업 시간 |
| diner_address_constituency | 행정구역 |
| all_review_cnt | 전체 리뷰 수 |
| real_good_review_cnt | 긍정적 리뷰 수 |
| real_bad_review_cnt | 부정적 리뷰 수 |
| real_good_review_percent | 긍정적 리뷰 비율 |
| real_bad_review_percent | 부정적 리뷰 비율 |

## 새로 추가/변경된 컬럼
- `diner_tag` 음식점별 특징, 예를 들어 혼밥, 혼술, 점심시간, 이동약자편의, 제로페이, 삼성페이 등 
- real_good_review_cnt 기준 -> [블로그 포스팅 참조](https://learningnrunning.github.io/post/tech/review/2024-10-03-What-to-eat-today/#%EC%A3%BC%EC%9A%94-%ED%8A%B9%EC%A7%95)

# review_df

- review_df_20241107_071929_yamyam_1.csv
- review_df_20241107_071929_yamyam_2.csv
- 수집된 리뷰 수: 559,268 개
  
| 컬럼명 | 설명 |
|---------|------|
| review_id | 리뷰 고유 식별자 |
| reviewer_id | 리뷰어 고유 식별자 |
| reviewer_review_cnt | 리뷰어가 작성한 총 리뷰 수 |
| reviewer_avg | 리뷰어의 평균 평점 |
| reviewer_review_score | 해당 리뷰의 평점 |
| reviewer_review | 리뷰 내용 |
| reviewer_review_date | 리뷰 작성 날짜 |
| reviewer_user_name | 리뷰어 닉네임/사용자명 |
| badge_grade | 리뷰어 배지 등급 |
| badge_level | 리뷰어 배지 레벨 |
| diner_idx | 음식점 고유 식별자 |
| reviewer_collected_review_cnt | 리뷰어의 수집된 리뷰 수 |

## 새로 추가/변경된 컬럼

- `badge_grade`와 `badge_level`은 리뷰어의 활동성과 신뢰도를 나타내는 지표
  - ![](https://blog.kakaocdn.net/dn/LbARw/btsczLDWyrc/zp7bToMZOCvFYj2iMJFWck/img.png)
- `reviewer_user_name`은 리뷰어의 식별 가능한 공개 이름(이전 reviewer_id)
- `review_id`는 각 리뷰를 구분하는 고유한 식별자
- `reviewer_collected_review_cnt`는 해당 리뷰어의 실제로 수집된 리뷰 수를 나타냄