from tkinter import filedialog

from paddleocr import PaddleOCR, draw_ocr
import re

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

FilePath = filedialog.askopenfilename()

# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
ocr = PaddleOCR(use_angle_cls=True, lang="ch",enable_mkldnn=True)  # need to run only once to download and load model into memory，enable_mkldnn在intel处理器上开启后可以提高速度

result = ocr.ocr(FilePath, cls=True)
print(result)

for line in result:
    text=re.findall(r"'(.*?)'",str(line))

print(str.join(' ',text))
# 显示结果



from PIL import Image
image = Image.open(FilePath).convert('RGB')
boxes = [detection[0] for line in result for detection in line] # Nested loop added
txts = [detection[1][0] for line in result for detection in line] # Nested loop added
scores = [detection[1][1] for line in result for detection in line] # Nested loop added
im_show = draw_ocr(image, boxes, txts, scores,  font_path='./fonts/font.ttf')
im_show = Image.fromarray(im_show)
im_show.save('test.jpg')
im_show.show()