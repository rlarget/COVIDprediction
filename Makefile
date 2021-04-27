run:
	scp Lung_Opacityvalid.xlsx hossain@best-linux.cs.wisc.edu:/u/h/o/hossain/stat453/final
all:
	scp -r ./final hossain@best-linux.cs.wisc.edu:/u/h/o/hossain/stat453

local:
	scp hossain@best-linux.cs.wisc.edu:/u/h/o/hossain/stat453/final/models/pretrained.pt .

