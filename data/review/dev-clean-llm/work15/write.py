import json
M={'IT':'công_nghệ_thông_tin_kỹ_thuật_số','DL':'du_lịch_nhà_hàng_khách_sạn_dịch_vụ','GD':'giáo_dục_đào_tạo_nghiên_cứu','KD':'kinh_doanh_bán_hàng_chăm_sóc_khách_hàng','DT':'kỹ_thuật_điện_điện_tử_viễn_thông','LG':'logistics_vận_tải_chuỗi_cung_ứng','MK':'marketing_truyền_thông_quảng_cáo_nội_dung','NN':'ngôn_ngữ_dịch_thuật','NS':'nhân_sự_hành_chính_pháp_chế_tư_vấn','KH':'nhóm_nghề_khác','NL':'nông_nghiệp_năng_lượng_môi_trường','SX':'sản_xuất_lao_động_phổ_thông_cơ_khí','TK':'thiết_kế_nghệ_thuật_giải_trí_truyền_hình_báo_chí','TC':'tài_chính_kế_toán_ngân_hàng_bảo_hiểm','XD':'xây_dựng_kiến_trúc_bất_động_sản','YT':'y_tế_dược_chăm_sóc_sức_khỏe_công_nghệ_sinh_học'}
dec={}
for l in open('work15/dec.txt',encoding='utf-8'):
    r,b,a,y=l.rstrip('\n').split('|')
    assert int(r) not in dec
    al=a.split(','); assert b in al
    dec[int(r)]=(M[b],[M[x] for x in al],y)
d=json.load(open('batch15.json'))
assert set(dec)==set(x['row'] for x in d), (set(x['row'] for x in d)-set(dec), set(dec)-set(x['row'] for x in d))
out=[{'row':x['row'],'nhan_tot_nhat':dec[x['row']][0],'cac_nhan_hop_ly':dec[x['row']][1],'ly_do':dec[x['row']][2]} for x in d]
json.dump(out,open('out15.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
print(len(out))
