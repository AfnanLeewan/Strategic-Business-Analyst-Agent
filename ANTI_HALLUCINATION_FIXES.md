# Anti-Hallucination Improvements for StratAI

## ปัญหาที่พบ (Problems Identified)

จากการทดสอบพบว่า StratAI มีปัญหา **Hallucination** หลายประการ:

### 1. ตัวเลขไม่ถูกต้อง (Incorrect Numbers)
- ❌ Electronics Sales: รายงาน $2.1M แต่ข้อมูลจริง $2.595M
- ❌ Furniture Sales: รายงาน $1.5M แต่ข้อมูลจริง $1.657M

### 2. การจัดอันดับภูมิภาคผิดพลาด (Wrong Regional Ranking)
- ❌ อ้างว่า South ทำผลงานแย่สุด แต่จริงๆ South อันดับ 2
- ❌ ภูมิภาคที่แย่สุดจริงๆ คือ West และ East

### 3. สินค้าขายดีผิด (Wrong Top Product)
- ❌ อ้างว่า Smartphone X ขายดีสุด
- ✅ สินค้าที่ขายดีจริง: Laptop Pro ($1.15M vs $681K)

### 4. สร้างข้อมูลเปรียบเทียบที่ไม่มี (Fabricated Comparisons)
- ❌ อ้างว่ามี Growth +12% เมื่อเทียบกับ Q1 2023
- ❌ อ้างว่า Electronics เติบโต +15%
- ❌ CSV มีข้อมูลเฉพาะ Q1 2024 ไม่มีข้อมูลเปรียบเทียบ

---

## การแก้ไข (Solutions Implemented)

### 1. ปรับปรุง RAG Prompt (backend/services/rag_chain.py)

#### เพิ่มกฎ Anti-Hallucination แบบเข้มงวด:

```
⚠️ CRITICAL ANTI-HALLUCINATION RULES:

1. ONLY USE PROVIDED DATA
   - ใช้เฉพาะตัวเลขที่มีใน Context เท่านั้น
   - ห้ามสร้างสถิติ เปอร์เซ็นต์ หรือการเปรียบเทียบขึ้นมาเอง
   - ถ้าไม่มีข้อมูล ต้องบอกว่า "Data not available"

2. NO FABRICATED COMPARISONS
   - ห้ามเปรียบเทียบกับปีก่อน ไตรมาสก่อน ถ้าไม่มีข้อมูล
   - ห้ามอ้าง "growth of X%" ถ้าไม่มีข้อมูล baseline
   - แทนที่จะบอกว่า "เติบโต +15%" ให้บอกว่า "ยอดขายปัจจุบัน $X"

3. EXACT NUMBER CITATION
   - รายงานตัวเลขที่ตรงกับในข้อมูล
   - ถ้าข้อมูลบอก 2,595,000 ห้ามปัดเป็น 2.1M (ต่างกัน 20%)
   - อนุญาตให้ปัดเศษเพื่อความอ่านง่าย แต่ต้องอยู่ในเกณฑ์ ±5%

4. VERIFY BEFORE CLAIMING
   - Top seller: ต้องตรวจสอบสินค้าทั้งหมดก่อนบอกว่าอันไหนอันดับ 1
   - Regional ranking: ต้องเปรียบเทียบทุกภูมิภาคก่อนบอกว่าที่ไหนดีสุด/แย่สุด
   - หมวดหมู่: ต้องรวมตัวเลขที่ให้มาจริงๆ

5. STATE LIMITATIONS EXPLICITLY
   - ถ้าไม่มีข้อมูลย้อนหลัง: "Note: ไม่สามารถเปรียบเทียบ growth ได้"
   - ถ้าข้อมูลไม่ครบ: "Based on available data for [ช่วงเวลา]"
```

#### เพิ่มส่วน Data Limitations:
- บังคับให้ AI ต้องระบุข้อจำกัดของข้อมูลทุกครั้ง
- เพิ่มส่วน "Data Limitations" ในรายงาน

### 2. ปรับปรุงการสรุปข้อมูล CSV (backend/services/ingest.py)

#### เพิ่มรายละเอียดที่ครบถ้วน:

**เพิ่มการระบุช่วงเวลา:**
```python
- Date Range: 2024-01-15 to 2024-02-16
⚠️ NOTE: This dataset contains ONLY data for the period above. 
         NO historical comparison data available.
```

**เพิ่มการแยกข้อมูลตามหมวดหมู่:**
```
BREAKDOWN BY CATEGORY
Category: Electronics
  - Number of records: 28
  - Total Sales: 2,595,000.00
  - Total Units: 1,022
  
Category: Furniture
  - Number of records: 21
  - Total Sales: 1,657,000.00
  - Total Units: 438
```

**เพิ่มการจัดอันดับสินค้า:**
```
TOP PRODUCTS BY SALES
1. Laptop Pro: 1,150,000.00
2. Smartphone X: 681,000.00
3. Monitor 4K: 602,000.00
...
```

**เพิ่มคำเตือนเกี่ยวกับข้อจำกัดข้อมูล:**
```
⚠️ IMPORTANT DATA LIMITATIONS
- This dataset represents a SNAPSHOT for the period shown
- NO historical comparison data available
- Growth rates CANNOT be calculated from this data alone
- Analysis should focus on current period performance only
```

---

## ผลลัพธ์ที่คาดหวัง (Expected Results)

หลังจากการแก้ไข StratAI ควรจะ:

### ✅ รายงานตัวเลขที่ถูกต้อง
- Electronics: $2.595M (ไม่ใช่ $2.1M)
- Furniture: $1.657M (ไม่ใช่ $1.5M)

### ✅ จัดอันดับถูกต้อง
- Top Product: Laptop Pro (ไม่ใช่ Smartphone X)
- Regions: North > South > East > West (ถูกต้อง)

### ✅ ไม่สร้างการเปรียบเทียบที่ไม่มีข้อมูล
- ไม่อ้างว่า "growth +12%" ถ้าไม่มีข้อมูลปีก่อน
- ไม่อ้างว่า "decline -2%" ถ้าไม่มีข้อมูลเปรียบเทียบ

### ✅ ระบุข้อจำกัดของข้อมูล
- บอกชัดเจนว่าข้อมูลครอบคลุมช่วงเวลาไหน
- บอกชัดเจนว่าไม่สามารถคำนวณ growth ได้เพราะไม่มีข้อมูลย้อนหลัง

---

## วิธีทดสอบ (How to Test)

1. **อัพโหลดไฟล์ CSV ใหม่:**
   - ลบ vector store เดิม (ถ้ามี)
   - อัพโหลด company_sales_q1_2024.csv ใหม่

2. **ถามคำถามเดียวกัน:**
   ```
   "วิเคราะห์แนวโน้มยอดขาย"
   ```

3. **ตรวจสอบผลลัพธ์:**
   - ✅ ตัวเลขต้องตรงกับข้อมูลจริง (±5%)
   - ✅ ไม่มีการอ้าง growth/decline % ถ้าไม่มีข้อมูลเปรียบเทียบ
   - ✅ สินค้าขายดีต้องเป็น Laptop Pro
   - ✅ ภูมิภาค South ต้องอันดับ 2 (ไม่ใช่แย่สุด)
   - ✅ มีส่วน "Data Limitations" บอกข้อจำกัด

---

## หมายเหตุสำคัญ (Important Notes)

### การอัพโหลดไฟล์ใหม่
เนื่องจากข้อมูลที่ ingest ไปแล้วยังเป็นแบบเก่า คุณต้อง:
1. Stop backend server (Ctrl+C)
2. ลบ vectordb folder (ถ้ามี)
3. Start backend server ใหม่
4. อัพโหลดไฟล์ CSV ใหม่

### LLM Model ที่ใช้
- ถ้าใช้ Ollama (llama3.1:8b): อาจจะยังมี hallucination เล็กน้อย
- ถ้าใช้ GPT-4: จะดีกว่า แต่ต้องมี OpenAI API key
- Prompt ที่ปรับปรุงจะช่วยลด hallucination ได้มาก แต่ไม่ได้หมดเลย 100%

### การปรับแต่งเพิ่มเติม
ถ้ายังมี hallucination อาจต้อง:
1. เพิ่ม temperature ให้ต่ำลง (0.3-0.5)
2. เพิ่ม retrieval chunks (k=15-20)
3. ใช้ Re-ranking model
4. เพิ่ม Few-shot examples ในตัวอย่างการตอบ

---

## สรุป (Summary)

การแก้ไขนี้เน้นที่:
1. **Prompt Engineering**: เพิ่มกฎที่เข้มงวดห้าม hallucination
2. **Better Context**: ให้ข้อมูลที่ละเอียดและครบถ้วนมากขึ้น
3. **Explicit Warnings**: บอกชัดเจนถึงข้อจำกัดของข้อมูล

ควรจะลด hallucination ลงได้มาก แต่อาจไม่หมด 100% เพราะเป็นข้อจำกัดของ LLM เอง
