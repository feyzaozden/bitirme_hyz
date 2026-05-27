"""
convert_to_teknofest_json.py

Bu modül, eğitilmiş YOLO modelinin tahminlerini TEKNOFEST Havacılıkta Yapay Zeka
yarışmasının nesne tespit çıktısına benzer JSON formatına dönüştürmek için yazılmıştır.

Üretilen her JSON çıktısında:
- cls
- landing_status
- motion_status
- top_left_x
- top_left_y
- bottom_right_x
- bottom_right_y

alanları yer almaktadır.
"""