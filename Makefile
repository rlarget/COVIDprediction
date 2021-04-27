#Easily move files from remote to local, if using cloud computing
run:
	scp Lung_Opacityvalid.xlsx hossain@cloud.computer:/path/to/wherever

#Move folder to remote
all:
	scp -r ./folder hossain@cloud.computer:/path/to/wherever
	
#Move files to local
local:
	scp hossain@cloud.computer:/path/to/wherever/models/pretrained.pt .

