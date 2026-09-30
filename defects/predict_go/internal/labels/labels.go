package labels

type Item struct {
	Name       string `json:"name"`
	CNName     string `json:"cn_name"`
	IsDefect   bool   `json:"is_defect"`
	InTrainSet bool   `json:"in_train_set"`
}

var Catalog = []Item{
	{Name: "GlueSeam", CNName: "胶缝", IsDefect: true, InTrainSet: true},
	{Name: "Below", CNName: "封边过低", IsDefect: true, InTrainSet: true},
	{Name: "Above", CNName: "封边过高", IsDefect: true, InTrainSet: false},
	{Name: "Tackless", CNName: "开胶", IsDefect: true, InTrainSet: true},
	{Name: "Shorter", CNName: "短带", IsDefect: true, InTrainSet: true},
	{Name: "Longer", CNName: "过长", IsDefect: true, InTrainSet: true},
	{Name: "Longer2", CNName: "过长2", IsDefect: true, InTrainSet: true},
	{Name: "Scratch", CNName: "划伤", IsDefect: true, InTrainSet: true},
	{Name: "Scrape", CNName: "刮板", IsDefect: true, InTrainSet: true},
	{Name: "GlueResidue", CNName: "残胶", IsDefect: true, InTrainSet: true},
	{Name: "GlueGap", CNName: "崩缺", IsDefect: true, InTrainSet: true},
	{Name: "BandBroken", CNName: "封边带损伤", IsDefect: true, InTrainSet: true},
	{Name: "BandWave", CNName: "波浪纹", IsDefect: true, InTrainSet: false},
	{Name: "Bumps", CNName: "鼓包", IsDefect: true, InTrainSet: true},
	{Name: "ClampGlue", CNName: "夹胶", IsDefect: true, InTrainSet: true},
	{Name: "ResidualTape", CNName: "残带", IsDefect: false, InTrainSet: true}, // 屏蔽非缺陷
	{Name: "Hole", CNName: "钻孔", IsDefect: false, InTrainSet: true},
	{Name: "Zigzag", CNName: "锯齿", IsDefect: false, InTrainSet: true},
	{Name: "Label", CNName: "标签", IsDefect: false, InTrainSet: true},
}

var byName = func() map[string]Item {
	m := make(map[string]Item, len(Catalog))
	for _, x := range Catalog {
		m[x.Name] = x
	}
	return m
}()

func CNNameOf(name string) string {
	if item, ok := byName[name]; ok {
		return item.CNName
	}
	return ""
}

func Payload() []Item {
	out := make([]Item, len(Catalog))
	copy(out, Catalog)
	return out
}
