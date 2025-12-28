// ============================================================
// CatRAG Sample Data - Week 1 Task A3
// ============================================================
// Purpose: Load sample data for UIT courses, departments, programs
// Created: November 14, 2025
// Owner: Team A - Infrastructure
// Note: This is SAMPLE data based on typical UIT structure
// ============================================================

// ============================================================
// 1. Create Departments (KHOA)
// ============================================================

CREATE (khoa_cntt:KHOA {
  ma_khoa: 'CNTT',
  ten_khoa: 'Khoa Công nghệ thông tin',
  ten_khoa_en: 'Faculty of Information Technology',
  website: 'https://fit.uit.edu.vn',
  email: 'fit@uit.edu.vn',
  phone: '028-3725-2002',
  truong_khoa: 'PGS.TS. Nguyễn Tấn Trần Minh Khang',
  created_at: datetime(),
  updated_at: datetime()
});

CREATE (khoa_kt:KHOA {
  ma_khoa: 'KT',
  ten_khoa: 'Khoa Kỹ thuật',
  ten_khoa_en: 'Faculty of Engineering',
  website: 'https://fe.uit.edu.vn',
  email: 'fe@uit.edu.vn',
  created_at: datetime(),
  updated_at: datetime()
});

CREATE (khoa_khtn:KHOA {
  ma_khoa: 'KHTN',
  ten_khoa: 'Khoa Khoa học tự nhiên',
  ten_khoa_en: 'Faculty of Natural Sciences',
  website: 'https://ns.uit.edu.vn',
  email: 'ns@uit.edu.vn',
  created_at: datetime(),
  updated_at: datetime()
});

// ============================================================
// 2. Create Academic Programs (CHUONG_TRINH_DAO_TAO)
// ============================================================

MATCH (khoa_cntt:KHOA {ma_khoa: 'CNTT'})
CREATE (ct_cntt:CHUONG_TRINH_DAO_TAO {
  ma_chuong_trinh: 'CNTT2023',
  ten_chuong_trinh: 'Công nghệ thông tin',
  khoa: 'CNTT',
  bac_dao_tao: 'Cử nhân',
  nam_bat_dau: 2023,
  tong_tin_chi: 140,
  mo_ta: 'Chương trình đào tạo kỹ sư Công nghệ thông tin, trang bị kiến thức nền tảng về lập trình, cấu trúc dữ liệu, cơ sở dữ liệu, mạng máy tính và các kỹ năng phát triển phần mềm',
  created_at: datetime(),
  updated_at: datetime()
})
CREATE (khoa_cntt)-[:QUAN_LY {created_at: datetime()}]->(ct_cntt);

MATCH (khoa_cntt:KHOA {ma_khoa: 'CNTT'})
CREATE (ct_ktpm:CHUONG_TRINH_DAO_TAO {
  ma_chuong_trinh: 'KTPM2023',
  ten_chuong_trinh: 'Kỹ thuật phần mềm',
  khoa: 'CNTT',
  bac_dao_tao: 'Cử nhân',
  nam_bat_dau: 2023,
  tong_tin_chi: 140,
  mo_ta: 'Chương trình đào tạo kỹ sư Kỹ thuật phần mềm, chuyên sâu về quy trình phát triển phần mềm, quản lý dự án, kiến trúc phần mềm và đảm bảo chất lượng',
  created_at: datetime(),
  updated_at: datetime()
})
CREATE (khoa_cntt)-[:QUAN_LY {created_at: datetime()}]->(ct_ktpm);

// ============================================================
// 3. Create Core Courses (MON_HOC) - Foundation Courses
// ============================================================

// IT001 - Nhập môn lập trình
MATCH (khoa_cntt:KHOA {ma_khoa: 'CNTT'})
CREATE (it001:MON_HOC {
  ma_mon: 'IT001',
  ten_mon: 'Nhập môn lập trình',
  ten_mon_en: 'Introduction to Programming',
  so_tin_chi: 4,
  khoa: 'CNTT',
  hoc_ky: 1,
  mo_ta: 'Giới thiệu các khái niệm cơ bản về lập trình máy tính, ngôn ngữ C/C++, cấu trúc điều khiển, hàm, mảng và chuỗi',
  noi_dung: 'Chương 1: Giới thiệu về lập trình\nChương 2: Biến và kiểu dữ liệu\nChương 3: Toán tử và biểu thức\nChương 4: Cấu trúc điều khiển\nChương 5: Hàm\nChương 6: Mảng\nChương 7: Chuỗi',
  created_at: datetime(),
  updated_at: datetime()
})
CREATE (it001)-[:THUOC_KHOA {created_at: datetime()}]->(khoa_cntt);

// IT002 - Lập trình hướng đối tượng
MATCH (khoa_cntt:KHOA {ma_khoa: 'CNTT'})
CREATE (it002:MON_HOC {
  ma_mon: 'IT002',
  ten_mon: 'Lập trình hướng đối tượng',
  ten_mon_en: 'Object-Oriented Programming',
  so_tin_chi: 4,
  khoa: 'CNTT',
  hoc_ky: 2,
  mo_ta: 'Học về các khái niệm lập trình hướng đối tượng: lớp, đối tượng, kế thừa, đa hình, đóng gói. Sử dụng ngôn ngữ C++/Java',
  noi_dung: 'Chương 1: Giới thiệu OOP\nChương 2: Lớp và đối tượng\nChương 3: Constructor và Destructor\nChương 4: Kế thừa\nChương 5: Đa hình\nChương 6: Đóng gói và trừu tượng hóa\nChương 7: Templates và Exceptions',
  created_at: datetime(),
  updated_at: datetime()
})
CREATE (it002)-[:THUOC_KHOA {created_at: datetime()}]->(khoa_cntt);

// IT003 - Cấu trúc dữ liệu và giải thuật
MATCH (khoa_cntt:KHOA {ma_khoa: 'CNTT'})
CREATE (it003:MON_HOC {
  ma_mon: 'IT003',
  ten_mon: 'Cấu trúc dữ liệu và giải thuật',
  ten_mon_en: 'Data Structures and Algorithms',
  so_tin_chi: 4,
  khoa: 'CNTT',
  hoc_ky: 3,
  mo_ta: 'Học về các cấu trúc dữ liệu cơ bản (danh sách, ngăn xếp, hàng đợi, cây, đồ thị) và các giải thuật sắp xếp, tìm kiếm',
  noi_dung: 'Chương 1: Phân tích giải thuật\nChương 2: Danh sách liên kết\nChương 3: Ngăn xếp và hàng đợi\nChương 4: Cây nhị phân\nChương 5: Cây tìm kiếm nhị phân\nChương 6: Heap và Priority Queue\nChương 7: Đồ thị\nChương 8: Sắp xếp và tìm kiếm',
  created_at: datetime(),
  updated_at: datetime()
})
CREATE (it003)-[:THUOC_KHOA {created_at: datetime()}]->(khoa_cntt);

// IT004 - Cơ sở dữ liệu
MATCH (khoa_cntt:KHOA {ma_khoa: 'CNTT'})
CREATE (it004:MON_HOC {
  ma_mon: 'IT004',
  ten_mon: 'Cơ sở dữ liệu',
  ten_mon_en: 'Database Systems',
  so_tin_chi: 4,
  khoa: 'CNTT',
  hoc_ky: 4,
  mo_ta: 'Kiến thức về thiết kế cơ sở dữ liệu, mô hình quan hệ, SQL, chuẩn hóa dữ liệu, transaction và quản trị CSDL',
  noi_dung: 'Chương 1: Giới thiệu CSDL\nChương 2: Mô hình ER\nChương 3: Mô hình quan hệ\nChương 4: SQL cơ bản\nChương 5: SQL nâng cao\nChương 6: Chuẩn hóa\nChương 7: Transaction và Concurrency\nChương 8: Indexing',
  created_at: datetime(),
  updated_at: datetime()
})
CREATE (it004)-[:THUOC_KHOA {created_at: datetime()}]->(khoa_cntt);

// IT005 - Nhập môn mạng máy tính
MATCH (khoa_cntt:KHOA {ma_khoa: 'CNTT'})
CREATE (it005:MON_HOC {
  ma_mon: 'IT005',
  ten_mon: 'Nhập môn mạng máy tính',
  ten_mon_en: 'Introduction to Computer Networks',
  so_tin_chi: 4,
  khoa: 'CNTT',
  hoc_ky: 4,
  mo_ta: 'Kiến thức về kiến trúc mạng, mô hình OSI/TCP-IP, giao thức mạng, địa chỉ IP, routing và switching',
  noi_dung: 'Chương 1: Tổng quan về mạng\nChương 2: Mô hình OSI\nChương 3: TCP/IP\nChương 4: Địa chỉ IP\nChương 5: Routing\nChương 6: Transport Layer\nChương 7: Application Layer',
  created_at: datetime(),
  updated_at: datetime()
})
CREATE (it005)-[:THUOC_KHOA {created_at: datetime()}]->(khoa_cntt);

// SE104 - Nhập môn công nghệ phần mềm
MATCH (khoa_cntt:KHOA {ma_khoa: 'CNTT'})
CREATE (se104:MON_HOC {
  ma_mon: 'SE104',
  ten_mon: 'Nhập môn công nghệ phần mềm',
  ten_mon_en: 'Introduction to Software Engineering',
  so_tin_chi: 4,
  khoa: 'CNTT',
  hoc_ky: 3,
  mo_ta: 'Giới thiệu về quy trình phát triển phần mềm, phân tích yêu cầu, thiết kế, kiểm thử và bảo trì phần mềm',
  noi_dung: 'Chương 1: Tổng quan CNPM\nChương 2: Quy trình phát triển\nChương 3: Phân tích yêu cầu\nChương 4: Thiết kế phần mềm\nChương 5: Kiểm thử\nChương 6: Quản lý dự án',
  created_at: datetime(),
  updated_at: datetime()
})
CREATE (se104)-[:THUOC_KHOA {created_at: datetime()}]->(khoa_cntt);

// SE357 - Phát triển ứng dụng web
MATCH (khoa_cntt:KHOA {ma_khoa: 'CNTT'})
CREATE (se357:MON_HOC {
  ma_mon: 'SE357',
  ten_mon: 'Phát triển ứng dụng web',
  ten_mon_en: 'Web Application Development',
  so_tin_chi: 4,
  khoa: 'CNTT',
  hoc_ky: 5,
  mo_ta: 'Học cách phát triển ứng dụng web với HTML, CSS, JavaScript, backend frameworks và cơ sở dữ liệu',
  noi_dung: 'Chương 1: HTML/CSS\nChương 2: JavaScript cơ bản\nChương 3: React/Vue\nChương 4: Node.js/Express\nChương 5: REST API\nChương 6: Authentication\nChương 7: Database Integration',
  created_at: datetime(),
  updated_at: datetime()
})
CREATE (se357)-[:THUOC_KHOA {created_at: datetime()}]->(khoa_cntt);

// SE363 - Trí tuệ nhân tạo
MATCH (khoa_cntt:KHOA {ma_khoa: 'CNTT'})
CREATE (se363:MON_HOC {
  ma_mon: 'SE363',
  ten_mon: 'Trí tuệ nhân tạo',
  ten_mon_en: 'Artificial Intelligence',
  so_tin_chi: 4,
  khoa: 'CNTT',
  hoc_ky: 6,
  mo_ta: 'Giới thiệu về trí tuệ nhân tạo, machine learning, deep learning, xử lý ngôn ngữ tự nhiên và computer vision',
  noi_dung: 'Chương 1: Giới thiệu AI\nChương 2: Tìm kiếm và tối ưu\nChương 3: Machine Learning cơ bản\nChương 4: Neural Networks\nChương 5: Deep Learning\nChương 6: NLP\nChương 7: Computer Vision',
  created_at: datetime(),
  updated_at: datetime()
})
CREATE (se363)-[:THUOC_KHOA {created_at: datetime()}]->(khoa_cntt);

// ============================================================
// 4. Create Prerequisites (DIEU_KIEN_TIEN_QUYET)
// ============================================================

// IT002 requires IT001
MATCH (it002:MON_HOC {ma_mon: 'IT002'}), (it001:MON_HOC {ma_mon: 'IT001'})
CREATE (it002)-[:DIEU_KIEN_TIEN_QUYET {
  loai: 'bat_buoc',
  diem_toi_thieu: 5.0,
  ghi_chu: 'Phải hoàn thành IT001 trước khi đăng ký IT002',
  created_at: datetime()
}]->(it001);

// IT003 requires IT002
MATCH (it003:MON_HOC {ma_mon: 'IT003'}), (it002:MON_HOC {ma_mon: 'IT002'})
CREATE (it003)-[:DIEU_KIEN_TIEN_QUYET {
  loai: 'bat_buoc',
  diem_toi_thieu: 5.0,
  ghi_chu: 'Cần nắm vững OOP trước khi học CTDL',
  created_at: datetime()
}]->(it002);

// IT004 requires IT003
MATCH (it004:MON_HOC {ma_mon: 'IT004'}), (it003:MON_HOC {ma_mon: 'IT003'})
CREATE (it004)-[:DIEU_KIEN_TIEN_QUYET {
  loai: 'bat_buoc',
  diem_toi_thieu: 5.0,
  ghi_chu: 'CTDL là nền tảng cho thiết kế CSDL',
  created_at: datetime()
}]->(it003);

// SE104 requires IT002
MATCH (se104:MON_HOC {ma_mon: 'SE104'}), (it002:MON_HOC {ma_mon: 'IT002'})
CREATE (se104)-[:DIEU_KIEN_TIEN_QUYET {
  loai: 'bat_buoc',
  diem_toi_thieu: 5.0,
  ghi_chu: 'Cần biết OOP để học CNPM',
  created_at: datetime()
}]->(it002);

// SE357 requires IT004
MATCH (se357:MON_HOC {ma_mon: 'SE357'}), (it004:MON_HOC {ma_mon: 'IT004'})
CREATE (se357)-[:DIEU_KIEN_TIEN_QUYET {
  loai: 'bat_buoc',
  diem_toi_thieu: 5.0,
  ghi_chu: 'Cần CSDL để phát triển web backend',
  created_at: datetime()
}]->(it004);

// SE357 requires SE104
MATCH (se357:MON_HOC {ma_mon: 'SE357'}), (se104:MON_HOC {ma_mon: 'SE104'})
CREATE (se357)-[:DIEU_KIEN_TIEN_QUYET {
  loai: 'khuyen_nghi',
  diem_toi_thieu: 5.0,
  ghi_chu: 'Nên có kiến thức CNPM cơ bản',
  created_at: datetime()
}]->(se104);

// SE363 requires IT003
MATCH (se363:MON_HOC {ma_mon: 'SE363'}), (it003:MON_HOC {ma_mon: 'IT003'})
CREATE (se363)-[:DIEU_KIEN_TIEN_QUYET {
  loai: 'bat_buoc',
  diem_toi_thieu: 5.0,
  ghi_chu: 'CTDL và giải thuật là nền tảng cho AI',
  created_at: datetime()
}]->(it003);

// ============================================================
// 5. Link Courses to Programs (THUOC_CHUONG_TRINH)
// ============================================================

MATCH (it001:MON_HOC {ma_mon: 'IT001'}), (ct:CHUONG_TRINH_DAO_TAO {ma_chuong_trinh: 'CNTT2023'})
CREATE (it001)-[:THUOC_CHUONG_TRINH {
  loai_mon: 'bat_buoc',
  hoc_ky_khuyen_nghi: 1,
  created_at: datetime()
}]->(ct);

MATCH (it002:MON_HOC {ma_mon: 'IT002'}), (ct:CHUONG_TRINH_DAO_TAO {ma_chuong_trinh: 'CNTT2023'})
CREATE (it002)-[:THUOC_CHUONG_TRINH {
  loai_mon: 'bat_buoc',
  hoc_ky_khuyen_nghi: 2,
  created_at: datetime()
}]->(ct);

MATCH (it003:MON_HOC {ma_mon: 'IT003'}), (ct:CHUONG_TRINH_DAO_TAO {ma_chuong_trinh: 'CNTT2023'})
CREATE (it003)-[:THUOC_CHUONG_TRINH {
  loai_mon: 'bat_buoc',
  hoc_ky_khuyen_nghi: 3,
  created_at: datetime()
}]->(ct);

MATCH (it004:MON_HOC {ma_mon: 'IT004'}), (ct:CHUONG_TRINH_DAO_TAO {ma_chuong_trinh: 'CNTT2023'})
CREATE (it004)-[:THUOC_CHUONG_TRINH {
  loai_mon: 'bat_buoc',
  hoc_ky_khuyen_nghi: 4,
  created_at: datetime()
}]->(ct);

MATCH (se357:MON_HOC {ma_mon: 'SE357'}), (ct:CHUONG_TRINH_DAO_TAO {ma_chuong_trinh: 'KTPM2023'})
CREATE (se357)-[:THUOC_CHUONG_TRINH {
  loai_mon: 'bat_buoc',
  hoc_ky_khuyen_nghi: 5,
  created_at: datetime()
}]->(ct);

MATCH (se363:MON_HOC {ma_mon: 'SE363'}), (ct:CHUONG_TRINH_DAO_TAO {ma_chuong_trinh: 'CNTT2023'})
CREATE (se363)-[:THUOC_CHUONG_TRINH {
  loai_mon: 'tu_chon',
  hoc_ky_khuyen_nghi: 6,
  created_at: datetime()
}]->(ct);

// ============================================================
// 6. Create Categories (CATEGORY)
// ============================================================

CREATE (cat_co_so:CATEGORY {
  name: 'CO_SO',
  description: 'Các môn học cơ sở ngành',
  parent_category: 'MON_HOC',
  created_at: datetime()
});

CREATE (cat_chuyen_nganh:CATEGORY {
  name: 'CHUYEN_NGANH',
  description: 'Các môn chuyên ngành',
  parent_category: 'MON_HOC',
  created_at: datetime()
});

CREATE (cat_tu_chon:CATEGORY {
  name: 'TU_CHON',
  description: 'Các môn tự chọn',
  parent_category: 'MON_HOC',
  created_at: datetime()
});

// ============================================================
// 7. Create Tags (TAG)
// ============================================================

CREATE (:TAG {name: 'lap_trinh', type: 'skill', created_at: datetime()});
CREATE (:TAG {name: 'co_so_du_lieu', type: 'skill', created_at: datetime()});
CREATE (:TAG {name: 'giai_thuat', type: 'skill', created_at: datetime()});
CREATE (:TAG {name: 'web_dev', type: 'skill', created_at: datetime()});
CREATE (:TAG {name: 'ai_ml', type: 'skill', created_at: datetime()});
CREATE (:TAG {name: 'de', type: 'difficulty', created_at: datetime()});
CREATE (:TAG {name: 'trung_binh', type: 'difficulty', created_at: datetime()});
CREATE (:TAG {name: 'kho', type: 'difficulty', created_at: datetime()});

// ============================================================
// 8. Tag Courses (TAGGED_AS)
// ============================================================

MATCH (it001:MON_HOC {ma_mon: 'IT001'}), (tag:TAG {name: 'lap_trinh', type: 'skill'})
CREATE (it001)-[:TAGGED_AS {created_at: datetime()}]->(tag);

MATCH (it001:MON_HOC {ma_mon: 'IT001'}), (tag:TAG {name: 'de', type: 'difficulty'})
CREATE (it001)-[:TAGGED_AS {created_at: datetime()}]->(tag);

MATCH (it003:MON_HOC {ma_mon: 'IT003'}), (tag:TAG {name: 'giai_thuat', type: 'skill'})
CREATE (it003)-[:TAGGED_AS {created_at: datetime()}]->(tag);

MATCH (it003:MON_HOC {ma_mon: 'IT003'}), (tag:TAG {name: 'kho', type: 'difficulty'})
CREATE (it003)-[:TAGGED_AS {created_at: datetime()}]->(tag);

MATCH (it004:MON_HOC {ma_mon: 'IT004'}), (tag:TAG {name: 'co_so_du_lieu', type: 'skill'})
CREATE (it004)-[:TAGGED_AS {created_at: datetime()}]->(tag);

MATCH (se357:MON_HOC {ma_mon: 'SE357'}), (tag:TAG {name: 'web_dev', type: 'skill'})
CREATE (se357)-[:TAGGED_AS {created_at: datetime()}]->(tag);

MATCH (se363:MON_HOC {ma_mon: 'SE363'}), (tag:TAG {name: 'ai_ml', type: 'skill'})
CREATE (se363)-[:TAGGED_AS {created_at: datetime()}]->(tag);

// ============================================================
// 9. Categorize Courses (THUOC_DANH_MUC)
// ============================================================

MATCH (it001:MON_HOC {ma_mon: 'IT001'}), (cat:CATEGORY {name: 'CO_SO'})
CREATE (it001)-[:THUOC_DANH_MUC {created_at: datetime()}]->(cat);

MATCH (it002:MON_HOC {ma_mon: 'IT002'}), (cat:CATEGORY {name: 'CO_SO'})
CREATE (it002)-[:THUOC_DANH_MUC {created_at: datetime()}]->(cat);

MATCH (it003:MON_HOC {ma_mon: 'IT003'}), (cat:CATEGORY {name: 'CO_SO'})
CREATE (it003)-[:THUOC_DANH_MUC {created_at: datetime()}]->(cat);

MATCH (se357:MON_HOC {ma_mon: 'SE357'}), (cat:CATEGORY {name: 'CHUYEN_NGANH'})
CREATE (se357)-[:THUOC_DANH_MUC {created_at: datetime()}]->(cat);

MATCH (se363:MON_HOC {ma_mon: 'SE363'}), (cat:CATEGORY {name: 'TU_CHON'})
CREATE (se363)-[:THUOC_DANH_MUC {created_at: datetime()}]->(cat);

// ============================================================
// 10. Create Sample Lecturers (GIANG_VIEN)
// ============================================================

MATCH (khoa_cntt:KHOA {ma_khoa: 'CNTT'})
CREATE (gv1:GIANG_VIEN {
  ma_gv: 'GV001',
  ten_gv: 'TS. Nguyễn Văn A',
  khoa: 'CNTT',
  email: 'nva@uit.edu.vn',
  chuyen_nganh: 'Cấu trúc dữ liệu và giải thuật',
  hoc_vi: 'Tiến sĩ',
  created_at: datetime(),
  updated_at: datetime()
});

MATCH (khoa_cntt:KHOA {ma_khoa: 'CNTT'})
CREATE (gv2:GIANG_VIEN {
  ma_gv: 'GV002',
  ten_gv: 'PGS.TS. Trần Thị B',
  khoa: 'CNTT',
  email: 'ttb@uit.edu.vn',
  chuyen_nganh: 'Trí tuệ nhân tạo',
  hoc_vi: 'Phó Giáo sư',
  created_at: datetime(),
  updated_at: datetime()
});

// Link courses to lecturers
MATCH (it003:MON_HOC {ma_mon: 'IT003'}), (gv1:GIANG_VIEN {ma_gv: 'GV001'})
CREATE (it003)-[:DUOC_DAY_BOI {
  hoc_ky: 'HK1_2024',
  vai_tro: 'giang_vien_chinh',
  created_at: datetime()
}]->(gv1);

MATCH (se363:MON_HOC {ma_mon: 'SE363'}), (gv2:GIANG_VIEN {ma_gv: 'GV002'})
CREATE (se363)-[:DUOC_DAY_BOI {
  hoc_ky: 'HK1_2024',
  vai_tro: 'giang_vien_chinh',
  created_at: datetime()
}]->(gv2);

// ============================================================
// Verification Queries
// ============================================================

// Count all nodes by label
// MATCH (n) RETURN labels(n)[0] as NodeType, count(*) as Count ORDER BY Count DESC;

// Show prerequisite chain for SE363
// MATCH path = (target:MON_HOC {ma_mon: 'SE363'})-[:DIEU_KIEN_TIEN_QUYET*]->(prereq)
// RETURN path;

// Show all courses in CNTT program
// MATCH (course:MON_HOC)-[r:THUOC_CHUONG_TRINH]->(program:CHUONG_TRINH_DAO_TAO {ma_chuong_trinh: 'CNTT2023'})
// RETURN course.ma_mon, course.ten_mon, r.loai_mon, r.hoc_ky_khuyen_nghi
// ORDER BY r.hoc_ky_khuyen_nghi;
