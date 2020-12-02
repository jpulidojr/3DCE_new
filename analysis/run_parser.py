import matplotlib.pyplot as plt


def Average(lst): 
    return sum(lst) / len(lst) 


#filename = "out.108049" #no compress
#filename = "out.110327" #olcf blosc, compress only
#filename = "out.110368" #my blosc, compress only
filename = "out.110551" #my blosc, compress+original


time=[]
timestep=[]
write_time=[] # Same as recall
comp_time=[]

timen=0

fio = open(filename, "r")
for line in fio:
    if 'Output file' in line:
        #print(line)
        s = line.split(' ')
        #print(s)
        s[-1] = s[-1].strip() #remove the \n
        #print(s)
        #print(s[8])
        timestep.append(float(s[8]))
        time.append(float(timen))
        timen=timen+1

        
    #Note: The validation criteria for 3DCE uses Sensitivity @ 4, in the log files.
    #Note2: Sensitivity is the same as Recall    
    if 'Write plotfile' in line:
        #print(line)
        s=line.split(' ')
        #print(s)
        s[-1] = s[-1].strip() #remove the \n
        #r = s[-1].split()
        #print(s[4])
        #print(r)
        write_time.append(float(s[4]))

    #Note: The validation criteria for 3DCE uses Sensitivity @ 4, in the log files.
    #Note2: Sensitivity is the same as Recall    
    if 'Allblosc' in line:
        #print(line)
        s=line.split(' ')
        #print(s)
        s[-1] = s[-1].strip() #remove the \n
        #r = s[-1].split()
        #print(s[7][:-1])
        comp_time.append(float(s[7][:-1]))


fio.close()

write_mean = Average(write_time) 
print(write_mean)
comp_mean = Average(comp_time) 
print(comp_mean)

#print(epoch)
#print(timestep)
#print(timestep_accumulated)
#print(write_time)
#print(epoch_precision)
#print(epoch_recall)
#print(epoch_fmeasure)

#plt.plot(epoch, write_time, epoch, epoch_precision, epoch, epoch_fmeasure)
plt.plot(timestep, write_time, label="Write Time" )
plt.plot(timestep, comp_time, label="Comp Time" )
#plt.plot(timestep, comp_time )
#plt.legend(['Write Time'])
#plt.legend(['Write Time'])
plt.ylim((0, 100))   
plt.ylabel('Time (seconds)')
plt.xlabel('Timestep')
plt.legend()
plt.show()


#plt.plot(timestep_accumulated, write_time)
#plt.ylabel('recall (sensitivity)')
#plt.xlabel('timestep (s)')
#plt.show()
