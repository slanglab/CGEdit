import os
import re

# This file will preprocess the CORAAL transcripts. It will remove
#   non-linguistic sounds and will split the transcribed speech into sentences
#   i.e. lines ending with a period, exclamation point, or question mark.
# The below variables define the directory where you are storing all 
#   CORAAL transcripts you'd like to preproces (orig_dir), and the directory where
#   you'd like the resulting preprocessed txt files to be stored (final_dir). 
orig_dir = "./CORAAL_transcripts/"
final_dir = "./CORAAL_preproc/"



# This is a list of regex to remove from the CORAAL transcripts.
#   Will remove non-linguistic sounds denoted by parentheses, angle brackets, or
#   backslashes e.g. (laughing), <cough>, /unintelligible/.
#   This will also remove the square brackets denoting overlapping speech while not
#   remove the overlapping speech itself.
toRemove = [r'\([^)]*\)', r'<[^>]*>', r'/unintelligible/', r'/inaudible/', 
        r'/[?]*/', r'[\[\]/]']
# These two lists of regex will help us split the transcribed speech into lines where
#   each line ends with a period, question mark, or exclamation point.
plist = [r'\.[ ]+', r'\?[ ]+', r'![ ]+']
rlist = ['.\n', '?\n', '!\n']

# Iterates through each file in the directory where we're storing the transcripts
with os.scandir(orig_dir) as d:
    for test_file in d:
        if test_file.name.startswith("."): continue
        first = True
        # Opens the file
        with open(test_file.path, encoding='utf8',errors='ignore') as r:
            # spkr1 is the interviewee; spkr2 is the interviewer
            spkr1 = ""
            spkr2 = ""
            # Iterates through each line in the file
            for line in r:
                # Skips the first line in the file, which is a header
                if first:
                    first = False
                    continue
                # Splits line into its components:
                #   'Line', 'Speaker', 'St Time, 'Content', and 'En Time'
                lineSplit = line.split("\t")
                # Here we take the 'Content' component, which is the transcribed speech
                sent = lineSplit[3]
                # If there is a string in the transcribed speech that matches any of 
                #   the regex defined in our list toRemove, it is removed
                for each in toRemove:
                    sent = re.sub(each, '', sent)
                # If the speaker of this line is the interviewee, add the speech to spkr1
                if lineSplit[1] in test_file.name[:-3]: spkr1 += " " + sent
                # If the speaker of this line is the interviewer, add to spkr2
                else: 
                    spkr2 += " " + sent
                    spkr2Name = lineSplit[1]
            # We now have all the transcribed speech in this file for each speaker 
            #   For each speaker, add a line break after each period, question mark, 
            #   or exclamation point
            for x in range(len(plist)):
                spkr1 = re.sub(plist[x], rlist[x], spkr1)
                spkr2 = re.sub(plist[x], rlist[x], spkr2)
            # For the interviewer, name the preprocessed file:
            #   INT-<interviewer speaker code>-<interviewee speaker code>.txt
            with open(final_dir + "INT-"+spkr2Name+"-"+test_file.name[:-3]+"txt",'a') as r:
                r.write(spkr2.strip())
            # For the interviewee, name the preprocessed file:
            #   <interviewee speaker code>.txt
            with open(final_dir + test_file.name[:-3]+"txt",'a') as r:
                r.write(spkr1.strip())
                


