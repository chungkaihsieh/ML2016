from PIL import Image
import sys
Lena = Image.open(sys.argv[1])
width, height = Lena.size
img = Image.new("RGB",(width,height),"white")
for x in range(width):
	for y in range(height):
		pix_value = Lena.getpixel((width-1-x, height-1-y))
		img.putpixel((x, y), (pix_value, pix_value, pix_value)) 
img.save("ans2.png")
