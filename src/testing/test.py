from PIL import Image, ImageDraw

img = Image.open("./uploads/78.png").convert("RGBA")
overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)

polygon = [
    (395, 258), (450, 278), (455, 295), (435, 345),
    (390, 340), (370, 320), (372, 285), (395, 258)
]

draw.polygon(polygon, fill=(0, 255, 0, 120))
draw.line(polygon + [polygon[0]], fill=(0, 200, 0, 255), width=2)

result = Image.alpha_composite(img, overlay)
result.save("78_corrected.png")