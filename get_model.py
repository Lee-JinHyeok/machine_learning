import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from PIL import Image
import requests
from io import BytesIO

def start():

    # 모델 불러오기
    model = tf.keras.models.load_model('/home/dasol/snap/food_person_ver/ver5/food_person_ver5.h5')
    # 모델 구조 출력
    # model.summary()
    url = ['https://scontent-ssn1-1.cdninstagram.com/v/t51.2885-15/sh0.08/e35/s640x640/118229895_1515833935278501_2230310252969095667_n.jpg?_nc_ht=scontent-ssn1-1.cdninstagram.com&_nc_cat=109&_nc_ohc=GLEdXZREdEgAX-xqGO1&oh=dd02ae6f64584716fa606240c51313e4&oe=5F6F4CBC',
           'https://scontent-ssn1-1.cdninstagram.com/v/t51.2885-15/sh0.08/e35/s640x640/118356474_659278394971071_6317306877636548981_n.jpg?_nc_ht=scontent-ssn1-1.cdninstagram.com&_nc_cat=103&_nc_ohc=NmUbFBuS5soAX8SuuCj&oh=5be19e8f128dfbab6c3d3c0beaf7aee1&oe=5F70C94A',
           'https://scontent-ssn1-1.cdninstagram.com/v/t51.2885-15/sh0.08/e35/s640x640/118172778_307756653824120_173857782520077920_n.jpg?_nc_ht=scontent-ssn1-1.cdninstagram.com&_nc_cat=108&_nc_ohc=vhZC4iEWpDgAX9RzjJk&oh=1b03b330aa91c03c8ba9e76f6c81aaef&oe=5F6F3976',
           'https://scontent-ssn1-1.cdninstagram.com/v/t51.2885-15/sh0.08/e35/s640x640/118172756_184542489774275_672655812953146618_n.jpg?_nc_ht=scontent-ssn1-1.cdninstagram.com&_nc_cat=111&_nc_ohc=tsnAan--4w8AX-RCkFo&oh=be439b42d2554abfde724826592f7c8c&oe=5F70DE24',
           'https://scontent-ssn1-1.cdninstagram.com/v/t51.2885-15/sh0.08/e35/s640x640/118276029_749272295862403_4208303411273838823_n.jpg?_nc_ht=scontent-ssn1-1.cdninstagram.com&_nc_cat=107&_nc_ohc=oaDcP794AvUAX8nSCSV&oh=62299d5fde836431fe6403908f231993&oe=5F708D67',
           'https://scontent-ssn1-1.cdninstagram.com/v/t51.2885-15/sh0.08/e35/s640x640/118098781_179192780403937_1172881386120259399_n.jpg?_nc_ht=scontent-ssn1-1.cdninstagram.com&_nc_cat=101&_nc_ohc=ZQU-8MKJFXMAX9OLZV5&oh=d78fa4d398750fc3c45f761d62684ec2&oe=5F6F6F4B',
           'https://scontent-ssn1-1.cdninstagram.com/v/t51.2885-15/sh0.08/e35/c0.0.1439.1439a/s640x640/118312445_3916053141745416_3416528575642680452_n.jpg?_nc_ht=scontent-ssn1-1.cdninstagram.com&_nc_cat=100&_nc_ohc=ONRycaoCIzUAX_KgdTC&oh=19350ecda42c340e510e4a2d091f2e07&oe=5F6DA9A5',
           'https://scontent-ssn1-1.cdninstagram.com/v/t51.2885-15/sh0.08/e35/s640x640/118155880_3152139488198505_1099600633236268471_n.jpg?_nc_ht=scontent-ssn1-1.cdninstagram.com&_nc_cat=109&_nc_ohc=0kNtHpcZalMAX8rwHrA&oh=36718ded6c365d761066337571c898ba&oe=5F6F249E',

           'https://scontent-ssn1-1.cdninstagram.com/v/t51.2885-15/sh0.08/e35/s640x640/118284537_184531123108902_4194850730287753419_n.jpg?_nc_ht=scontent-ssn1-1.cdninstagram.com&_nc_cat=106&_nc_ohc=cpCWQ_glzzMAX82JjCd&oh=809d3951974b187d32433a882fc2f812&oe=5F6F2955',
           'https://scontent-ssn1-1.cdninstagram.com/v/t51.2885-15/sh0.08/e35/s640x640/118357182_669033783691779_543001307151296680_n.jpg?_nc_ht=scontent-ssn1-1.cdninstagram.com&_nc_cat=103&_nc_ohc=VHffyCczYekAX-omQhq&oh=f511861e4d70cb2bdff67e0ddd3ee800&oe=5F6EBD79',
           'https://scontent-ssn1-1.cdninstagram.com/v/t51.2885-15/sh0.08/e35/s640x640/118307351_797115791103727_6168387993216732033_n.jpg?_nc_ht=scontent-ssn1-1.cdninstagram.com&_nc_cat=107&_nc_ohc=CW9cFP1FkewAX-EdGPf&oh=55ff80ae59e5fa6d129a8f97460afb8e&oe=5F6DA512',
           'https://scontent-ssn1-1.cdninstagram.com/v/t51.2885-15/sh0.08/e35/c134.0.812.812a/s640x640/118297495_1262220330786336_2256130881004152286_n.jpg?_nc_ht=scontent-ssn1-1.cdninstagram.com&_nc_cat=100&_nc_ohc=63hcKNdgXywAX8ECKRD&oh=16e5aecd773bc1d0bf9dde308fdb5cc3&oe=5F70E78F',
           'https://scontent-ssn1-1.cdninstagram.com/v/t51.2885-15/sh0.08/e35/c180.0.1080.1080a/s640x640/118482198_170836444602475_8792785315706596497_n.jpg?_nc_ht=scontent-ssn1-1.cdninstagram.com&_nc_cat=107&_nc_ohc=hcGG3SzCsUkAX-nzI7z&oh=090f5b9c09889d8def14bb22291189d6&oe=5F6EAC2B'
           ]

    for link in url:
        # print(link)
        response = requests.get(link)
        img = Image.open(BytesIO(response.content))
        img = img.resize((150, 150))

        # 내 로컬에서 가져오기
        # path = '/home/dasol/snap/machine_learning/img/'
        # img = image.load_img(path, target_size=(150, 150))
        # print(img)

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])

        classes = model.predict(images, batch_size=10)

        if classes[0] > 0:
            print(" is a person")
        else:
            print(" is a food")


if __name__ == '__main__':

    start()