"""Đường học sâu: PhoBERT → dense → một đầu ra.

Hai bài toán chạy **riêng lẻ** trước (``category`` và ``salary``); phần đa nhiệm
gộp hai nhánh vào một mất mát đến sau, khi mỗi nhánh đã có mốc riêng để so.

Module ``text`` không phụ thuộc torch để test đường dữ liệu chạy được ở mọi máy.
"""
