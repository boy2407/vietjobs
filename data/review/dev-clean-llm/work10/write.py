import json
M=dict(IT='công_nghệ_thông_tin_kỹ_thuật_số',DL='du_lịch_nhà_hàng_khách_sạn_dịch_vụ',GD='giáo_dục_đào_tạo_nghiên_cứu',KD='kinh_doanh_bán_hàng_chăm_sóc_khách_hàng',DT='kỹ_thuật_điện_điện_tử_viễn_thông',LG='logistics_vận_tải_chuỗi_cung_ứng',MK='marketing_truyền_thông_quảng_cáo_nội_dung',NN='ngôn_ngữ_dịch_thuật',NS='nhân_sự_hành_chính_pháp_chế_tư_vấn',KH='nhóm_nghề_khác',NL='nông_nghiệp_năng_lượng_môi_trường',SX='sản_xuất_lao_động_phổ_thông_cơ_khí',TK='thiết_kế_nghệ_thuật_giải_trí_truyền_hình_báo_chí',TC='tài_chính_kế_toán_ngân_hàng_bảo_hiểm',XD='xây_dựng_kiến_trúc_bất_động_sản',YT='y_tế_dược_chăm_sóc_sức_khỏe_công_nghệ_sinh_học')
lab={}
for l in open('work10/lab.txt',encoding='utf-8'):
    l=l.strip()
    if not l: continue
    r,b,o,why=l.split('|')
    assert int(r) not in lab, r
    lab[int(r)]=(b,[x for x in o.split(',') if x],why)
d=json.load(open('batch10.json'))
rows=[x['row'] for x in d]
assert set(rows)==set(lab), (set(rows)-set(lab), set(lab)-set(rows))
out=[]
for r in rows:
    b,o,why=lab[r]
    out.append(dict(row=r,nhan_tot_nhat=M[b],cac_nhan_hop_ly=[M[b]]+[M[x] for x in o if x!=b],ly_do=why))
json.dump(out,open('out10.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
print(len(out))
