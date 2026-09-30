package infer

import "sort"

func NMS(dets []Det, iouThresh float64) []Det {
	if len(dets) == 0 {
		return dets
	}
	order := make([]int, len(dets))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(i, j int) bool {
		return dets[order[i]].Score > dets[order[j]].Score
	})
	keep := make([]Det, 0, len(dets))
	for len(order) > 0 {
		i := order[0]
		keep = append(keep, dets[i])
		remain := order[:0]
		for _, j := range order[1:] {
			same := dets[i].Name == dets[j].Name
			if same && iouXYXY(dets[i].XYXY, dets[j].XYXY) >= iouThresh {
				continue
			}
			remain = append(remain, j)
		}
		order = remain
	}
	return keep
}

func iouXYXY(a, b [4]float64) float64 {
	ix1 := max(a[0], b[0])
	iy1 := max(a[1], b[1])
	ix2 := min(a[2], b[2])
	iy2 := min(a[3], b[3])
	iw := max(0.0, ix2-ix1)
	ih := max(0.0, iy2-iy1)
	inter := iw * ih
	if inter <= 0 {
		return 0
	}
	areaA := max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
	areaB := max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
	union := areaA + areaB - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}
