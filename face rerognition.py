import face_recognition as fr
import cv2

image = fr.load_image_file(r"آدرس تصویر اصلی ")

# پیدا کردن لوکیشن صورت در عکس
locations = fr.face_locations(image)
# علامت زدن صورت در تصویر 
bgrimg = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
bgrimg = cv2.rectangle(
    bgrimg,
    (locations[0][3], locations[0][0]),   # (left, top)
    (locations[0][1], locations[0][2]),   # (right, bottom)
    (255, 0, 0),
    3
)



sabt_sorat=fr.face_encodings(image)[0]

# خواندن صورت تست
image_test=fr.load_image_file(r"آدرس تصویری که مورد مقایسه قرار میگیرد")
image_test_encoding=fr.face_encodings(image_test)[0]


# مقایسه کردن دو تصویر 
moqayeseh=fr.compare_faces([sabt_sorat],image_test_encoding)



# نمایش تصویر 
cv2.imshow('image', bgrimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(locations)





# آیا دو عکس شبیه هم هست ؟
print(moqayeseh)
