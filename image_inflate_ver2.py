from keras.preprocessing.image import ImageDataGenerator, img_to_array
from PIL import Image
import requests, json
from datetime import datetime
import os
from io import BytesIO

def instagram_image_download():
    jpg_name = str(datetime.today().strftime("%m%d%H"))
    save_date = str(datetime.today().strftime("%Y-%m-%d-%H"))
    try:
        os.mkdir('/home/dasol/snap/crontab/instagram_image_data/person_inflate/' + save_date)
    except:
        print('이미 폴더가 존재')

    i, j, k = 0, 0, 0
    for keyword in keyword_list():
        print(keyword, ' : start')

        # print('::: 키워드 - {} url 크롤링 시작 :::'.format(keyword))
        # 해당 페이지 전체 원문 소스 크롤링
        r = ''
        r = requests.get('https://www.instagram.com/explore/tags/{}/?hl=ko'.format(keyword))
        # 전체 소스에서 데이터를 가지고 있는 부분만 잘라냄
        data = ''
        data = r.text.split('window._sharedData = ')[1].split(';</script>')[0]
        # data안의 가비지 데이터를 제외하고 json형식으로 저장
        json_obj = ''
        json_obj = json.loads(data)['entry_data']['TagPage'][0]['graphql']['hashtag']['edge_hashtag_to_media']['edges']

        # 사진 유형 만들기 변수
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        for meta in json_obj:
            text = str(meta['node']['accessibility_caption'])
            url = meta['node']['display_url']
            j += 1
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img = img.resize((150, 150))
            x = img_to_array(img)  # (3, 150, 150) 크기의 NumPy 배열
            x = x.reshape((1,) + x.shape)  # (1, 3, 150, 150) 크기의 NumPy 배열
            num = 0
            for batch in datagen.flow(x, batch_size=1,
                                      save_to_dir='/home/dasol/snap/crontab/instagram_image_data/person_inflate/{}'.format(save_date),
                                      save_prefix='btc_{}{}.jpg'.format(jpg_name, j), save_format='jpg'):
                num += 1
                if num > 20:
                    break

def keyword_list():
    keyword_list = ['인스타그램', '좋아요', '맞팔', '좋튀']
    return keyword_list


instagram_image_download()