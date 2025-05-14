import numpy as np
import pandas as pd
from io import StringIO
import torch
import os

import py_midicsv as pm #https://github.com/timwedde/py_midicsv
from midi_player import MIDIPlayer #https://pypi.org/project/midi-player/

def play_midi_file(path):
  return MIDIPlayer(path,200)

def df_to_pymidi(df):

    df = df.copy()

    df["Start"] = df["Start"].cumsum()
    df["Event"] = " Note_on_c"
    df["Track"] = 1
    df["Channel"] = 0
    df["Order"]=1

    df_stop = df.copy()
    df_stop["Event"] = " Note_off_c"
    df_stop["Start"] = df_stop["Start"] +  df_stop["Duration"]
    df_stop["Order"]=0

    df = pd.concat([df,df_stop],axis=0)
    df = df.reset_index()
    df = df.sort_values(["Start","Order"],axis=0,ascending=[True, True])

    df = df[["Track","Start","Event","Channel","Note","Volume"]]
    df = df.rename(columns={"Start":"Time"})

    header = ['0, 0, Header, 0, 1, 384\n',
            '1, 0, Start_track\n',
            '1, 0, Time_signature, 4, 2, 24, 8\n', #Track, Time, Time_signature, Num, Denom, Ticks per beat, NotesQ
            '1, 0, Tempo, 645161\n', #Number of microseconds per quarter note
            '1, 0, Title_t, "Elec. Piano (Classic)"\n',
            '1, 0, Program_c, 0, 0\n']

    csv_buffer = StringIO()
    df.to_csv(csv_buffer,header=False,index=False)
    event_list = csv_buffer.getvalue().split("\n")
    if(event_list[-1]==""):
        event_list = event_list[:-1]
    event_list = [x.replace(".0", "")+"\n" for x in event_list]

    maxT = df["Time"].iloc[-1]
    end_track = [f'1, {maxT:d}, End_track\n']

    song = header + event_list + end_track + ['0, 0, End_of_file']

    midi_object = pm.csv_to_midi(song)

    return midi_object

def pymidi_to_midi_file(midi_object, out_file):

    with open(out_file, "wb") as output_file:
        midi_writer = pm.FileWriter(output_file)
        midi_writer.write(midi_object)

def df_to_midi_file(df, out_file):
    midi_object = df_to_pymidi(df)
    pymidi_to_midi_file(midi_object, out_file)

def torch_to_df(X):
    df = pd.DataFrame({"Start":X[:,0].numpy(), "Duration":X[:,1].numpy(),"Note":X[:,2].numpy(),"Volume":X[:,3].numpy()})
    return df

def pt_to_df(file_pt):
    X = torch.load(file_pt)
    df = torch_to_df(X)
    return df

def torch_to_midi_file(X,out_file):
    df = torch_to_df(X)
    df_to_midi_file(df,out_file)

def play_torch(X):
    torch_to_midi_file(X,"temp.mid")
    return(play_midi_file("temp.mid"))

def play_pt_file(file):
    X = torch.load(file)
    return(play_torch(X))
