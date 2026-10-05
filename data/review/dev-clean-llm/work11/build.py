import json
M={'IT':'công_nghệ_thông_tin_kỹ_thuật_số','DL':'du_lịch_nhà_hàng_khách_sạn_dịch_vụ','GD':'giáo_dục_đào_tạo_nghiên_cứu','KD':'kinh_doanh_bán_hàng_chăm_sóc_khách_hàng','DIEN':'kỹ_thuật_điện_điện_tử_viễn_thông','LOG':'logistics_vận_tải_chuỗi_cung_ứng','MKT':'marketing_truyền_thông_quảng_cáo_nội_dung','NN':'ngôn_ngữ_dịch_thuật','NS':'nhân_sự_hành_chính_pháp_chế_tư_vấn','KHAC':'nhóm_nghề_khác','NONG':'nông_nghiệp_năng_lượng_môi_trường','SX':'sản_xuất_lao_động_phổ_thông_cơ_khí','TK':'thiết_kế_nghệ_thuật_giải_trí_truyền_hình_báo_chí','TC':'tài_chính_kế_toán_ngân_hàng_bảo_hiểm','XD':'xây_dựng_kiến_trúc_bất_động_sản','YT':'y_tế_dược_chăm_sóc_sức_khỏe_công_nghệ_sinh_học'}
dec={}
for i in range(1,7):
  for l in open(f'd{i}.txt',encoding='utf-8'):
    l=l.strip()
    if not l: continue
    r,b,a,why=l.split('|')
    assert int(r) not in dec, r
    al=a.split(','); assert b in al
    dec[int(r)]=dict(nhan_tot_nhat=M[b],cac_nhan_hop_ly=[M[x] for x in al],ly_do=why)
inp=json.load(open('../batch11.json'))
out=[dict(row=x['row'],**dec[x['row']]) for x in inp]
assert len(out)==len(inp)==len(dec)
json.dump(out,open('../out11.json','w'),ensure_ascii=False,indent=1)
print(len(out))
