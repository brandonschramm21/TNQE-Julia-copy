import os

initial_dist = 0.5
final_dist = 2
stepsize = 0.05

bondlength_list = ""

for i in range(int((final_dist - initial_dist) / stepsize) + 1):
    
    dist = initial_dist + i * stepsize
    formatted = "{:.2f}".format(dist)
    with open(str(formatted)+".xyz", "w") as file:
        file.write("2\n\n")
        file.write("H "+str(formatted)+" 0 0\n")
        file.write("H 0 0 0\n")
    os.system("mv "+str(formatted)+".xyz ../configs/xyz_files/h2_stretch/")

    bondlength_list = bondlength_list + formatted + ","
print(bondlength_list)
