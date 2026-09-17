# PTZ pixel-error to motion relation

Đây là phần mô hình hiện tại. Xem [README.md](README.md) để có hướng dẫn đầy
đủ về cài đặt, fit, lập manifest và chạy camera thật.

`build_ptz_pixel_motion_relation.py` dùng 30 H đã lưu làm cầu nối qua hệ Wide
cố định; không nội suy H theo pan/tilt. Với tâm `c` của PTZ pose đích `j`:

```text
w_j = H_j c
p_i = H_i⁻¹ w_j
e_i = c - p_i
```

Nhãn tương ứng là:

```text
Δpan  = pan_j  - pan_i
Δtilt = tilt_j - tilt_i
```

Script tạo hai bộ dữ liệu:

- `ptz_pair_transfer_samples.*`: chuyển các điểm lưới bất kỳ giữa các pose;
- `ptz_motion_relation_samples.*`: mẫu đưa điểm về tâm, dùng để fit model.

Model vận hành là Polynomial bậc 2 không intercept:

```text
ex = (centre_u - u) / (width / 2)
ey = (centre_v - v) / (height / 2)

[Δpan, Δtilt] = [ex, ey, ex², ex·ey, ey²] B
```

`B` được tìm bằng least squares. Không có intercept vì điểm ở đúng tâm phải
cho delta bằng 0. Lệnh live cộng delta vào `source_actual_position`, không cộng
vào trạng thái yêu cầu nếu camera chưa đạt trạng thái đó.

Tất cả pose hiện tại có `zoom=0`, nên model chưa học quan hệ theo zoom. Test
centre-only cố ý không dùng SIFT/tracker ở online; ảnh đích chỉ đánh dấu tâm PTZ,
vì vậy không thể tự đo sai số pixel của vật thể sau khi quay.
