#!/usr/bin/env python

"""Interactive client written for Open Pixel Control
http://github.com/zestyping/openpixelcontrol

Author: Keith Maki

To simulate:
First start the gl simulator using the unionjack descriptor

    make
    bin/gl_server layouts/unionjack.json

Then run this script in another shell to send colors to the simulator

    python_clients/typewriter.py

This client can be used to send signals to any openpixelcontrol-enabled server

Usage:
    python_clients/typewriter.py [ip:port]

"""

from __future__ import division
import time
import math
import sys, pdb
import select
import termios
import copy
import opc
import color_utils
from collections import namedtuple
from termcolor import colored,cprint

N_DIGITS = 1
N_DIGITS = 8  # digits in all the display rows
ROWS = [0,0,0,0,1,1,1,1] 
ORIENTATION =  [True, False]
N_PIXELS = 107 # pixels per digit
FPS = 5       # frames per second

#-------------------------------------------------------------------------------
# handle command line

if len(sys.argv) == 1:
    IP_PORT = '10.42.0.56:7890'
elif len(sys.argv) == 2 and ':' in sys.argv[1] and not sys.argv[1].startswith('-'):
    IP_PORT = sys.argv[1]
else:
    print
    print '    Usage: typewriter.py [ip:port]'
    print
    print '    If not set, ip:port defauls to 127.0.0.1:7890'
    print
    sys.exit(0)


#-------------------------------------------------------------------------------
# connect to server

client = opc.Client(IP_PORT)
if client.can_connect():
    print '    connected to %s' % IP_PORT
else:
    # can't connect, but keep running in case the server appears later
    print '    WARNING: could not connect to %s' % IP_PORT
print

#-------------------------------------------------------------------------------
# define character outputs
class UnionJack:

    fcs = [55]
    lc  = [7]
    rc  = [31]
    tc  = [19]
    bc  = [43]
    trr = [80]
    brr = [106]
    tlr = [63]
    blr = [89]
    tid = fcs+[56]
    tjd = [60,61]
    bng = [48,49]
    krn = range(17,22)
    
    lb  = range( 0,  7)
    lbh = range( 0,  5)
    lu  = range( 8, 15)
    lhs = range( 0, 15)
    ltd = range( 7,  9)
    
    tl  = range(15, 19)
    tr  = range(20, 24)
    tlh = range(16, 19)
    trh = range(20, 23)
    trp = range(16, 24)
    tlp = range(15, 23)
    top = range(15, 24)
    
    ru  = range(24, 31)
    rb  = range(32, 39)
    rbh = range(34, 39)
    rhs = range(24, 39)
    
    br  = range(39, 43)
    bl  = range(44, 48)
    brh = range(40, 43)
    blh = range(44, 47)
    brp = range(39, 47)
    blp = range(40, 48)
    bls = range(41, 48)
    bot = range(39, 48)
    
    cb  = range(48, 55)
    cbh = range(48, 53)
    cbc = range(51, 55)
    cbp = range(48, 57)
    cu  = range(56, 63)
    cuc = range(56, 60)
    cup = range(53, 63)
    cc  = cbc+cuc+fcs
    cnt = range(48, 63)
    
    vl  = range(63, 72)
    vr  = range(72, 81)
    vee = range(63, 81)+fcs
    
    mr  = range(81, 85)
    ml  = range(85, 89)
    mid = range(81, 89)+fcs
    mcs = range(83, 87)
    
    kl  = range(89, 98)
    kr  = range(98, 107)
    krt = range(89, 107)+fcs
    
    bsh = vl+kr+fcs
    lsh = vr+kl+fcs
    cmc = cbh+[44]
    cmr = rbh+brr

    biUpper = range(9,15)+range(57,63)+range(24,30)+range(15,24)+range(63,71)+range(73,81)
    biLower = range(0,5)+range(48,54)+range(33,39)+range(99,107)+range(89,97)+range(39,47)

    _pixels = dict(zip(range(107),rhs+bot+lhs+top+cnt[::-1]+kr[::-1]+kl[::-1]+ml[::-1]+mr[::-1]+vr[::-1]+vl[::-1]))

    #string.ascii_letters+string.punctuation+string.digits
    lut = {"A":set(lhs+rhs+mid+top+trr+tlr),\
           "B":set(cnt+mr+ru+rb\
                   +trr+brr+br+tr\
                   +trp+brp),\
           "C":set(tlp+blp+lhs+tlr+blr),\
           "D":set(trp+brp+rhs+cnt\
                   +trr+brr),\
           "E":set(top+bot+lhs+ml\
                   +tlr+blr),\
           "F":set(top+ml+lhs+tlr),\
           "G":set(top+bot+lhs+rb+mr+tlr+blr+brr),\
           "H":set(lhs+rhs+mid),\
           "I":set(top+bot+cnt),\
           "J":set(lb+bl+cnt+krn+blr),\
           "K":set(lhs+cnt+vr+kr+fcs),\
           "IK":set(cnt+lhs+vr+kr+fcs),\
           "L":set(lhs+bot+blr),\
           "LI":set([el-107*4 for el in range(428,435)+range(460,467)+range(471,476)+range(517,518)]),\
           "M":set(lhs+rhs+vee),\
           "N":set(lhs+rhs+bsh),\
           "O":set(top+bot+lhs+rhs+trr+brr+tlr+blr),\
           "P":set(top+mid+lhs+ru),\
           "Q":set(top+bot+lhs+rhs+kr),\
           "R":set(lhs+top+mid+ru+kr+trr+tlr)-set([14]),\
           "S":set(top+bot+mid+lu+rb+brr+trr+blr+tlr),\
           "SI":set([el-107 for el in range(122,126)+range(131,139)+range(171,179)+range(192,196)]),\
           "T":set(top+cnt+bc),\
           "Ts":set([el-107*5 for el in range(583,591)+range(616,624)]),\
           "U":set(lhs+rhs+bot+brr+blr),\
           "V":set(lhs+lsh),\
           "VI":set(range(24,32)+range(55,63)+range(64,72)),\
           "W":set(lhs+rhs+krt),\
           "X":set(vee+krt),\
           "Y":set(vee+cb+bc),\
           "Ys":set([el-107*6 for el in range(674,689)+range(740,749)]),\
           "Z":set(top+bot+lsh),\
           "a":set(bls+lb+cb+ml+blr)-set(bc),\
           "b":set(lhs+bl+ml+cb),\
           "c":set(ml+lb+bl+blr),\
           "d":set(lb+cnt+ml+bl),\
           "e":set(kl+ml+lb+blp+blr),\
           "f":set(cnt+mid+tr),\
           "g":set(tl+bl+ml+lu+cnt),\
           "h":set(lhs+cb+ml),\
           "i":set(cbh+bc+tid),\
           "ik":set(cnt+vr+kr+lbh+ltd),\
           "j":set(lbh+blr+bl+cbp+tjd),\
           "k":set(cnt+vr+kr),\
           "l":set(cnt),\
           "m":set(mid+lb+cb+rb),\
           "n":set(lb+cb+ml),\
           "o":set(lb+ml+cb+bl),\
           "p":set(lhs+tl+ml+cu),\
           "q":set(tl+ml+lu+cnt),\
           "r":set(lb+ml),\
           "s":set(tl+bl+ml+lu+cb+blr+tlr+[62]),\
           "t":set(cnt+mid+br+bc),\
           "u":set(lb+cb+bl+blr),\
           "v":set(lb+kl),\
           "w":set(lb+krt+rb),\
           "x":set(krt+vee+mid),\
           "y":set(cu+rhs+brp+mr+brr),\
           "z":set(ml+kl+bl),\
           "1":set(vr+rhs),\
           "16":set(lhs+tr+mr+br+cnt+rb),\
           "17":set(lhs[3:]+trp[3:]+tr+rhs[:-3]),\
               #"17":set(lu+tr+vr),\
           "2":set(top+bot+mid+ru+lb),\
           "2r":set(tr+ru+br+mr+cb),\
           "3":set(top+bot+mr+rhs),\
           "4":set(mid+rhs+lu),\
           "5":set(top+mid+bot+lu+rb),\
           "6":set(top+mid+bot+lhs+rb),\
           "7":set(top+lsh+mcs),\
           "8":set(top+bot+mid+rhs+lhs),\
           "9":set(top+mid+rhs+bot+lu),\
           "0":set(top+bot+rhs+lhs+lsh),\
           "!":set(cup+bng+tc),\
           "@":set(top+lhs+ru+bot+mr+vr+blr+trr+tlr),\
           "#":set(cnt+rhs+lbh+ltd),\
           "$":set(cnt+top+bot+mid+lu+rb+brr+trr+blr+tlr),\
           "%":set(tl+tlr+lu+cnt+mid+rb+brr+br+lsh),\
           "^":set(krt),\
           "&":set(tl+bsh+ml+cu+lb+bot+blr+brr+[37]),\
           "*":set(top+bot+lhs+rhs+trr+tlr+blr+brr+cnt+mid+lsh+bsh),\
           "(":set(vr+kr+[84]),\
           ")":set(vl+kl+[85]),\
           "-":set(mid),\
           "_":set(bot),\
           "+":set(mid+cc),\
           "=":set(top+mid),\
           "`":set(vl),\
           "~":set(lu+vee),\
           "{":set(tr+cu+cb+ml+br),\
           "}":set(tl+cu+cb+mr+bl),\
           "[":set(tr+br+cnt+bc+tc),\
           "]":set(tl+bl+cnt+bc+tc),\
           "\\":set(bsh),\
           "|":set(cnt+tc+bc),\
           ":":set(tid+bng),\
           ";":set(tjd+kl),\
           '"':set(cu+ru),\
           "'":set(cu),\
           "<":set(vr+kr+fcs),\
           ">":set(vl+kl+fcs),\
           ",":set(cmc),\
           ".":set(bc),\
           "/":set(lsh),\
           "?":set(tlr+top+trr+ru+mr+cbc+bc),\
           " ":set()
         }

    fliplut = {False:dict([(i,i) for i in range(107)]),\
               True:dict([(i,o) for (i,o) in zip(lhs+top+rhs+bot+cnt+vl+vr+mr+ml+kl+kr,\
                                           rhs+bot+lhs+top+cnt[::-1]+kr[::-1]+kl[::-1]\
                                           +ml[::-1]+mr[::-1]+vr[::-1]+vl[::-1])])}

    @classmethod
    def convert(cls,char,flip=False):
        return set([cls.fliplut[flip][el] for el in cls.lut[char]])

class SuperCycle():
    def __init__(self,indexable):
        self.curr = 0
        self.items = indexable
    
    def next(self):
        item = self.items[self.curr]
        self.curr = (self.curr + 1) % len(self.items)
        return item

    def prev(self):
        item = self.items[(self.curr - 2) % len(self.items)]
        self.curr = (self.curr - 1) % len(self.items)
        return item

    def insert(self,item):
        self.items.insert(self.curr,item)
        self.curr += 1

    def __getitem__(self, i):
        return self.items[i]

    def __len__(self):
        return len(self.items)
Color = namedtuple("Color","red, green, blue")

class ColorBanner():
    """
    Class to organize colored text.
    Currently supports static colors,
    and only one color per character
    """

    RED = Color(255,0,0)
    BLUE = Color(0,0,255)
    GREEN = Color(0,255,0)
    CYAN = Color(0,127,127)
    YELLOW = Color(200,127,0)
    MAGENTA = Color(127,0,127)
    PURPLE = Color(127,0,255) 
    PINK = Color(255,0,127)
    WHITE = Color(255,200,170)
    palette = SuperCycle([RED,
                          YELLOW,
                          GREEN,
                          CYAN,
                          BLUE,
                          MAGENTA,
                          WHITE])
        
    def __init__(self):
        self.text = []
        self.colors = []
        self.color = self.RED
        self.ind = 0

    def add(self,char):
        """Add char to the banner in self.color"""
        if self.length()<N_DIGITS:
            self.text = self.text[:self.ind]+[char]+self.text[self.ind:]
            self.colors = self.colors[:self.ind]\
                          +[self.color]+self.colors[self.ind:]
            self.ind += 1

    def bksp(self):
        """Delete the character just prior to the cursor, if any"""
        if self.ind>0:
            self.text = self.text[:self.ind-1]\
                        +self.text[self.ind:]
            self.colors = self.colors[:self.ind-1]\
                          +self.colors[self.ind:]
            self.ind -= 1

    def cursor_right(self):
        """Move the cursor right one position"""
        self.ind = min(self.ind+1,len(self.text))
           
    def cursor_left(self):
        """Move the cursor left one position"""
        self.ind = max(self.ind-1,0)

    def cycle_color_forward(self):
        """Set the cursor color to the next color on the palette"""
        self.color = self.palette.next()

    def cycle_color_backward(self):
        """Set the cursor color to the previous color on the palette"""
        self.color = self.palette.prev()        

    def delete(self):
        """Delete the character at the cursor, if any"""
        self.text = self.text[:self.ind]+self.text[self.ind+1:]
        self.colors = self.colors[:self.ind]+self.colors[self.ind+1:]

    def length(self):
        if self.text=="SIGBOV":
            return 8
        if self.text=="BIvis":
            return 7
        if self.text=="Jill2016":
            return 7
        return len(self.text)

    def getIndex(self):
        return self.ind

    def getIndices(self,digit):
        """Return the UnionJack pixel indices used by the banner text"""
        if self.text=="SIGBOV":
            return UnionJack.convert(['S','I','G','B','O','V','IK','17'][digit],\
                                     ORIENTATION[ROWS[digit]])
        if self.text=="Jill2016":
            return UnionJack.convert(['J','#',' ','2','*','1','6',' '][digit],\
                                     ORIENTATION[ROWS[digit]])
        elif self.text=="BIvis":
            return UnionJack.convert(['VI','SI','B','I','LI','Ts','Ys',' '][digit],\
                                     ORIENTATION[ROWS[digit]])
        else:
            return UnionJack.convert(self.text[digit],ORIENTATION[ROWS[digit]])

    def getPixel(self,digit,index):
        """Returns the color of the banner pixel at index in digit"""
        if self.text=="SIGBOV":
            if digit<3:
                return self.colors[digit]
            if digit<7:
                return self.colors[3]
            return self.WHITE
        if self.text[digit]=="*":
            return self.GREEN if index in UnionJack.lut['O'] else self.YELLOW 
        elif self.text=="Jill2016":
            if digit==1:
                return self.YELLOW if index in set(UnionJack.fliplut[True][el] for el in UnionJack.ltd) else self.GREEN
            elif digit==4:
                return self.GREEN if index in UnionJack.lut['O'] else self.YELLOW
            else:
                return self.GREEN
        elif self.text=="BIvis":
            if digit==2 or digit==3:
                return self.BLUE if index in set(UnionJack.biUpper) else (self.PINK if index in UnionJack.biLower else self.PURPLE)
            elif digit < 2:
                return self.WHITE
            else:
                return self.WHITE 
        else:
            return self.colors[digit]

    def substring(self,start,end):
        """
        Returns a new banner with the same colors,
        but only for the characters between start (inclusive)
        and end (exclusive)
        """
        banner = ColorBanner()
        banner.text = self.text[start:end]
        banner.colors = self.colors[start:end]
        banner.color = self.color
        banner.ind = min(max(self.ind-start,0),len(banner.text))
        banner.palette = copy.copy(self.palette)
        return banner

    def textify(self,char,color):
        def color_func(clr):
            if clr.red and clr.green and clr.blue:
                return lambda(x): colored(x,'white',attrs=['reverse'])
            if clr.red and clr.green:
                return lambda(x): colored(x,'yellow',attrs=['reverse'])
            if clr.red and clr.blue:
                return lambda(x): colored(x,'white','on_magenta')
            if clr.green and clr.blue:
                return lambda(x): colored(x,'cyan',attrs=['reverse'])
            if clr.red:
                return lambda(x): colored(x,'red')
            if clr.green:
                return lambda(x): colored(x,'green',attrs=['reverse'])
            if clr.blue:
                return lambda(x): colored(x,'white','on_blue')
            return lambda(x): x
        return color_func(color)(char)

    def toText(self):
        """
        Returns a printable version of the banner 
        using ANSI color text formatting
        """
        return "".join([self.textify(char,clr)
                        for (char,clr) in zip(self.text,self.colors)])

#-------------------------------------------------------------------------------
# Set up to recognize keyboard input
class KeyPoller():
    __x_1b = "\x1b"
    __CSI = "["
    __UP = "A"
    __DOWN = "B"
    __RIGHT = "C"
    __LEFT = "D"
    __DEL = "3"
    __BACKSPACE = "\x7f"

    def __init__(self, state):
        self.banner = state
        self.history = SuperCycle([state])

    def __enter__(self):
        # Save the terminal settings
        self.fd = sys.stdin.fileno()
        self.new_term = termios.tcgetattr(self.fd)
        self.old_term = termios.tcgetattr(self.fd)

        # New terminal setting unbuffered
        self.new_term[3] = (self.new_term[3] & ~termios.ICANON & ~termios.ECHO)
        termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.new_term)

        return self

    def __exit__(self, type, value, traceback):
        termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)

    def poll(self):
        dr,dw,de = select.select([sys.stdin], [], [], 0)
        if not dr == []:
            return sys.stdin.read(1)
        return None

    def check(self, i):
        state = self.history[i]
        c = self.poll()
        if c is not None:
            state = self.handle(c,state)
            sys.stdout.write("\r{0}     ".format(self.banner.toText()))
            sys.stdout.write("\r{0}".format(self.banner.substring(0,self.banner.getIndex()).toText()))
            sys.stdout.flush()
        return state

    def handle(self,c,state):
        if c=='\n':
            state = self.banner
            self.history.insert(self.banner)
            self.banner = ColorBanner()
            print ""
        elif c=='\t':
            self.banner = self.history.prev()
        elif c in UnionJack.lut:
            self.banner.add(c)
        elif c==self.__BACKSPACE:
            self.banner.bksp()
        elif c==self.__x_1b:
            if sys.stdin.read(1)==self.__CSI:
                c = sys.stdin.read(1)
                if c==self.__RIGHT:
                    self.banner.cursor_right()
                elif c==self.__LEFT:
                    self.banner.cursor_left()
                elif c==self.__DOWN:
                    self.banner.cycle_color_forward()
                elif c==self.__UP:
                    self.banner.cycle_color_backward()
                elif c==self.__DEL:
                    sys.stdin.read(1)
                    self.banner.delete()
            else:
                return self.handle(c,state)
        else:
            pdb.set_trace()
        return state

#-------------------------------------------------------------------------------
# send pixels
print '    sending pixels forever (control-c to exit)...'
print

banner = ColorBanner()
banner.color = banner.RED
banner.add('S')
banner.color = banner.GREEN
banner.add('I')
banner.color = banner.BLUE
banner.add('G')
banner.color = banner.YELLOW
banner.add('B')
banner.add('O')
banner.add('V')
banner.add('IK')
banner.color = banner.WHITE
banner.add('17')
start_time = time.time()
timestep = 0
with KeyPoller(banner) as keyPoller:
    while True:
        nextstep = time.time() + 1/FPS
        pixels = []
        msglen = banner.length()
        for rr in [0,1]:
            for ii in [dig for dig in range(N_DIGITS)[::(-1 if ORIENTATION[rr] else 1)] if rr==ROWS[dig]]:
                if ii<msglen:
                    indices = banner.getIndices(ii)
                for jj in range(107):
                    if ii<msglen and jj in indices:
                        pix = banner.getPixel(ii,jj)
                        rgb = (pix.red,pix.green,pix.blue)
                    else:
                        rgb = (0,0,0)
                    pixels.append(rgb)
        client.put_pixels(pixels, channel=0)
        while time.time() < nextstep:
            banner = keyPoller.check(timestep)
        if time.time() >= nextstep:
            nextstep += 1
            if len(keyPoller.history) > 1:
                timestep = (timestep + 1) % (len(keyPoller.history)-1)
