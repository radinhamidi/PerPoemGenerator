import numpy as np
import os

path_shahname = './dataset/shahname/'
path_boostan = './dataset/boostan/'
path_ghazal = './dataset/divan-ghazal/'

path = path_shahname

file_list = os.listdir(path)

def getCharText():
    text = ''
    for dir in file_list:
        with open(file=path+dir, mode="r", encoding="utf8") as file:
            text = text + '\n' + file.read()
    return text

def getWordText():
    text = getCharText()
    words = text.split(" ")
    return words