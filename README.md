# HSI Data Annotation (COCO-HSI Data Annotation)

โปรเจคนี้เป็น Desktop GUI แอปสำหรับสร้างและแก้ไข Ground Truth Annotation ของ Hyperspectral Image (HSI) แบบ pixel-level
โฟกัสในงาน classification mask และส่งออกทั้งเป็นไฟล์ภาพและ COCO-style JSON

## ✅ ไฮไลต์ฟีเจอร์หลัก

- โหลด hyperspectral datacube แบบ ENVI (`.hdr`) แล้วประมวลผล preview RGB
- คำนวณ band RGB จาก metadata หรือ default bands (R=29, G=19, B=9) และปรับด้วย percentile stretch
- ปักหมุด (paint) แยก layer/tag รองรับ class ID, class name, color ปรับแต่งได้
- Tools:
  - Cursor (move/resize / lock/unlock layer)
  - Connect/Polygon fill (v4 path closure)
  - Circle drawing
  - Pen (freehand), Eraser
  - Flood fill (ใช้งานอยู่ใน mode connect ใน canvas)
- เลือก class ผ่าน dropdown toolbar และ class table (ซ่อนจาก UI ปกติ แต่เรียกใช้งานได้)
- ดู spectrum ที่ cursor -> plot พร้อม class-avg spectra (รันใน thread เพื่อไม่ล็อก UI)
- ปรับ RGB contrast ด้วย low/high percentile (`2.0 - 98.0` ค่า default)
- บันทึก output เป็น:
  - PNG/TIFF grayscale (ค่าพิกเซล=class ID, 0=background)
  - COCO JSON (annotations polygon จาก mask หรือ bbox fallback)

## 🧩 โครงสร้างไฟล์

- `__main__.py` - entry point (`python -m __main__`)
- `hsi_annotation/app.py` - PyQt5 application bootstrap
- `hsi_annotation/canvas.py` - drawing logic, layer/mask management, input event handling
- `hsi_annotation/data.py` - datacube loading, RGB preview, class spectra, COCO conversion
- `hsi_annotation/ui/window.py` - main window, toolbar, status bar, save/open dialogs
- `hsi_annotation/ui/class_table.py` - class label manager
- `hsi_annotation/ui/paint_view.py` - zoom/pan view
- `hsi_annotation/ui/pg_panel.py` - mask preview + spectrum plot
- `test_script` - ตัวอย่างงานและสคริปต์ตรวจสอบ COCO JSON เดโม (รวม `test_annotation.py`)

## 📦 ติดตั้ง

แนะนำ Python 3.7+ (ทดสอบกับ 3.10 ขึ้นไป)

```bash
pip install PyQt5 numpy spectral pyqtgraph
``` 

สำหรับ `test_script` (optional):

```bash
pip install matplotlib imageio
```

### Shortcuts ที่มีในระบบ

- `Ctrl+O` เปิดไฟล์ `.hdr`
- `Ctrl+N` ล้าง GT layer ทั้งหมด
- `Ctrl+S` บันทึกเป็น PNG/TIFF/JSON
- `Ctrl+Shift+C` สร้าง class ใหม่
- `L` เปลี่ยนไปเครื่องมือ Connect
- `C` เครื่องมือ Circle
- `T` เพิ่ม Tag (Layer) ใหม่
- `Ctrl+U` Cursor tool
- `Ctrl+=` / `Ctrl+-` / `Ctrl+0` Zoom in/out/reset
- `Ctrl+F` Fit
- `Ctrl+Wheel` Zoom (fast)
- ปุ่มเมาส์กลางลากเพื่อ pan

## 🗂️ รูปแบบไฟล์ที่รองรับ

- Input: ENVI datacube (`.hdr` + data file เช่น `.bip` / `.bsq` / `.bil`)
- Output:
  - PNG/TIFF mask (8-bit grayscale, pixel value = class ID)
  - COCO JSON (ใน `save` dialog เลือก `.json`)

## 📌 เกณฑ์ class

- class id ต้องมากกว่า 0 และไม่เกิน 255
- class id ซ้ำไม่ได้
- class id 0 = background / ไม่มี annotation
- class name ใส่ได้ ไม่จำกัด

## 💡 วิธีดูผล COCO JSON

- `build_coco_annotation_json_from_layers(...)` จาก layer data
- output JSON format ตาม COCO ใกล้เคียง:
  - `images`: id, file_name, width, height
  - `annotations`: id, image_id, category_id, bbox, area, segmentation
  - `categories`: id, name

`test_script/test_annotation.py` ตรวจสอบและเรนเดอร์ annotation เป็นรูปภาพ

## 🛠️ ปรับจูน

- ปรับ cutoff preview ด้วย Contrast dialog แล้วระบบบันทึกค่าที่เลือก
- ปรับ/ล็อก/ซ่อน layer ผ่าน panel
- ภายใต้ hood: band selection ถอดจาก metadata nm; ถ้าไม่มีจะใช้ default

## 🔎 ข้อควรระวังและ limitation

- โหมด `cursor` ให้ลากกล่องหรือปรับไซส์ได้ใน layer ที่มี bounding box
- `fill` mode ใน `CanvasItem` ทำงานบน mask ที่มี layer active
- ถ้าไม่มี OpenCV (`cv2`) การคำนวณ segmentation polygon จาก mask จะ fallback เป็น bbox rectangle
- หากไม่มี `wavelength` ใน metadata จะใช้ default SWIR/VIS lookup

## Resuts From Annotation
### Cube RGB
![image](https://github.com/qthailand/coco-hsi-data-annotation/blob/main/test_script/turmeric.png)
### Result
![image](https://github.com/qthailand/coco-hsi-data-annotation/blob/main/test_script/coco_ann_ann_1.png)
![image](https://github.com/qthailand/coco-hsi-data-annotation/blob/main/test_script/coco_ann_ann_2.png)
![image](https://github.com/qthailand/coco-hsi-data-annotation/blob/main/test_script/coco_ann_ann_3.png)
![image](https://github.com/qthailand/coco-hsi-data-annotation/blob/main/test_script/coco_ann_ann_4.png)