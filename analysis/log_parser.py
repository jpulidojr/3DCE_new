import matplotlib.pyplot as plt
import csv


log_path = "../logs/"

#filename = "log.traintest_03-18_17-08-23.3DCE 1 image 3 slice" #previous run, random 30%, no annealing, 10 epochs

#filename = "log.traintest_04-16_08-08-30.3DCE 1 image 3 slice" #Smaller test run, 10 epochs 10% sampling
#filename = "log.traintest_04-16_09-45-43.3DCE 1 image 3 slice" #Latest run with sim_annealing, 10% sampling
#filename = "log.traintest_04-29_14-25-41.3DCE 1 image 3 slice" #Unfinished run with sim_annealing, 15% sampling
filename = "log.traintest_07-08_13-45-07.3DCE 1 image 3 slice" #Unfinished run with sim_annealing, 25% then 15% rand sampling

#filename = "log.traintest_07-22_15-02-43.3DCE 1 image 3 slice" #sim_annealing working, 5% sample and ramps up +5%
#filename = "log.traintest_07-22_15-47-28.3DCE 1 image 3 slice" #sim_annealing working, 5% sample and ramps up +5%

#filename = "log.traintest_08-30_17-17-16.3DCE 1 image 3 slice" # No anealing, standard run

#filename = "log.traintest_08-05_16-25-58.3DCE 1 image 3 slice" # sim_annealing with additive process, 25% with 10% ramp up

#filename = "log.traintest_08-05_23-23-09.3DCE 1 image 3 slice"
#filename = "log.traintest_08-07_13-22-36.3DCE 1 image 3 slice" # newest
files = [   "log.traintest_08-30_17-17-16.3DCE 1 image 3 slice", 
            "log.traintest_07-08_13-45-07.3DCE 1 image 3 slice",    
            "log.traintest_08-05_16-25-58.3DCE 1 image 3 slice",
            "log.traintest_08-05_23-23-09.3DCE 1 image 3 slice",
            "log.traintest_08-07_13-22-36.3DCE 1 image 3 slice" ]

for filename in files:
    epoch=[]
    epoch_time=[]
    epoch_time_accumulated=[]
    epoch_sensitivity=[] # Same as recall
    epoch_precision=[]
    epoch_recall=[] # same as racall
    epoch_fmeasure=[]
    epochn=0

    fio = open(log_path+filename, "r")
    prev_line=""
    for line in fio:
        if 'Time cost' in line and 'Epoch' in line:
            #print(line)
            s = line.split('=')
            s[-1] = s[-1].strip() #remove the \n
            #print(s)
            epoch_time.append(float(s[1]))
            epoch.append(float(epochn))
            epochn=epochn+1

            if len(epoch_time_accumulated) == 0:
                epoch_time_accumulated.append(float(s[1]))
            else:
                epoch_time_accumulated.append(epoch_time_accumulated[-1]+float(s[1]))
            
        #Note: The validation criteria for 3DCE uses Sensitivity @ 4, in the log files.
        #Note2: Sensitivity is the same as Recall    
        if 'Sensitivity' in line:
            #print(line)
            s=line.split(':')
            s[-1] = s[-1].strip() #remove the \n
            r = s[-1].split()
            #print(r)
            epoch_sensitivity.append(float(r[3]))

        # The log file prints the results below the string, so query prev line
        if 'precision' in prev_line:
            s=line.split()
            epoch_precision.append(float(s[3]))
        if 'recall' in prev_line:
            s=line.split()
            epoch_recall.append(float(s[3]))
        if 'f-measure' in prev_line:
            s=line.split()
            epoch_fmeasure.append(float(s[3]))

        prev_line=line

    fio.close()

    #print(epoch)
    #print(epoch_time)
    #print(epoch_time_accumulated)
    #print(epoch_sensitivity)
    #print(epoch_precision)
    #print(epoch_recall)
    #print(epoch_fmeasure)

    #plt.plot(epoch, epoch_sensitivity, epoch, epoch_precision, epoch, epoch_fmeasure)
    #plt.plot(epoch, epoch_sensitivity )
    #plt.ylabel('recall (sensitivity)')
    #axes = plt.gca()
    #axes.set_ylim([0,1])
    #plt.xlabel('# of epochs')
    #plt.savefig( 'epoch/' + filename + "_num_epoch.png")
    #plt.show()


    plt.plot(epoch_time_accumulated, epoch_sensitivity)
    plt.ylabel('recall (sensitivity)')
    plt.xlabel('epoch_time (s)')
    axes = plt.gca()
    axes.set_xlim([0,20000])
    axes.set_ylim([0,1])
    #plt.savefig( 'time/'+ filename + "_time.png")


plt.legend(["Original Ru", "Delete and random annealing", "Additive random annealing", "Annealing without duplicates", "Latest" ])
plt.show()


with open( 'csv/'+filename+'.csv', 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(epoch)
     wr.writerow(epoch_time_accumulated)
     wr.writerow(epoch_sensitivity)