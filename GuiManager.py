import os

import tkinter.font as tkf
from tkinter import *
from tkinter import filedialog

import MovieEditor as me


class MainFrame(Frame):
    # filePath와 savePath가 변수설정

    def __init__(self, master):
        Frame.__init__(self, master)

        self.master = master
        self.master.title("FACESCAPE")
        self.pack(fill=BOTH, expand=True)

        # 빈공간
        emptyFrame = Frame(self)
        emptyFrame.pack(fill=X)
        nameFont = tkf.Font(size=15, weight="bold")
        emptyLabel = Label(emptyFrame, text="FACESCAPE", font=nameFont)
        emptyLabel.pack(pady=10)

        # 파일 위치
        fileFrame = Frame(self)
        fileFrame.pack(fill=NONE)
        fileDirLbl = Label(fileFrame, text="파일 위치: ", width=10, height=1)
        self.filePath = StringVar()
        fileDirInput = Entry(fileFrame, textvariable=self.filePath, width=40)
        fileDirButton = Button(fileFrame, text="Find", width=7, height=1, command=self.LoadMovieFile, repeatdelay=100)
        fileDirLbl.pack(side=LEFT, padx=8, pady=10)
        fileDirInput.pack(side=LEFT, padx=3, pady=10)
        fileDirButton.pack(side=LEFT, padx=4, pady=10)

        # 저장 위치
        saveFrame = Frame(self)
        saveFrame.pack(fill=NONE)
        saveDirLbl = Label(saveFrame, text="저장 위치: ", width=10, height=1)
        self.savePath = StringVar()
        saveDirInput = Entry(saveFrame, textvariable=self.savePath, width=40)
        saveDirButton = Button(saveFrame, text="Find", width=7, height=1, command=self.SetOutputDirectory,
                               repeatdelay=100)
        saveDirLbl.pack(side=LEFT, padx=8, pady=10)
        saveDirInput.pack(side=LEFT, padx=3, pady=10)
        saveDirButton.pack(side=LEFT, padx=4, pady=10)

        # 파일 이름
        saveNameFrame = Frame(self)
        saveNameFrame.pack(fill=NONE)
        saveNameLbl = Label(saveNameFrame, text="파일 이름: ", width=10, height=1)
        self.saveFileName = StringVar()
        self.saveFileName.set("Output")
        saveNameInput = Entry(saveNameFrame, textvariable=self.saveFileName, width=20)
        saveNameLbl.pack(side=LEFT, padx=8, pady=10)
        saveNameInput.pack(side=LEFT, padx=3, pady=10)

        # 옵션 선택
        optionFrame = Frame(self)
        optionFrame.pack(fill=NONE)
        self.optionNumber = IntVar()
        option2Button = Radiobutton(optionFrame, text="얼굴 모자이크", value=1, variable=self.optionNumber)
        option2Button.pack(side=LEFT, anchor=CENTER, padx=4, pady=1)
        option1Button = Radiobutton(optionFrame, text="프레임 새기기", value=2, variable=self.optionNumber)
        option1Button.pack(side=LEFT, anchor=CENTER, padx=4, pady=1)
        option2Button = Radiobutton(optionFrame, text="프레임 삭제", value=3, variable=self.optionNumber)
        option2Button.pack(side=LEFT, anchor=CENTER, padx=4, pady=1)

        # 실행 버튼
        activateFrame = Frame(self)
        activateFrame.pack(fill=NONE)
        self.isActivated = False
        activateButton = Button(activateFrame, text="실행", width=7, height=1, command=self.Activate, repeatdelay=100)
        activateButton.pack(padx=4, pady=3)

        # 실행 상태
        progressFrame = Frame(self)
        progressFrame.pack(fill=BOTH)
        self.progressMessage = StringVar()
        progressLabel = Label(progressFrame, textvariable=self.progressMessage, relief=SOLID, bd=1,
                              bg='ghost white', width=50, height=30)
        progressLabel.pack(side=TOP, anchor=N, padx=5, pady=5)

    def LoadMovieFile(self):
        path = filedialog.askopenfilename(
            initialdir="C:",
            filetypes=([('Video Files(mkv, avi, mp4, mpg, flv, wmv)', '*.mkv;*.avi;*.mp4;*.mpg;*.flv;*.wmv')]))
        self.filePath.set(path)

    def SetOutputDirectory(self):
        path = filedialog.askdirectory(
            initialdir="C:")
        self.savePath.set(path)

    def Activate(self):
        checkFileMsg = me.checkFile(self.filePath.get())
        checkDirMsg = me.checkDirectory(self.savePath.get())

        if checkFileMsg is not None:
            self.SetProgressMessage(checkFileMsg)

        elif checkDirMsg is not None:
            self.SetProgressMessage(checkDirMsg)

        elif self.saveFileName.get() == "" or None:
            self.SetProgressMessage("파일 이름이 설정되지 않았습니다.")

        elif self.optionNumber.get() < 1 or self.optionNumber.get() > 3:
            self.SetProgressMessage("옵션이 선택되지 않았습니다.")

        else:
            self.progressMessage.set("실행 중")
            saveFilePath = self.savePath.get() + "/" + self.saveFileName.get()
            extension = me.FindExtension(self.filePath.get())
            me.Editing_Movie(self.filePath.get(), self.optionNumber.get(), saveFilePath, extension)
            self.progressMessage.set("실행 완료")
            os.startfile(self.savePath.get())

    def SetProgressMessage(self, msg):
        self.progressMessage.set(msg)


def MainGUI():
    root = Tk()

    screenWidth = root.winfo_screenwidth()
    screenHeight = root.winfo_screenheight()
    width = 480
    height = 320
    pos_x = int(screenWidth / 2 - width / 2)
    pos_y = int(screenHeight / 2 - height / 2)
    root.geometry(str(width) + "x" + str(height) + "+" + str(pos_x) + "+" + str(pos_y))

    MainFrame(root)

    root.resizable(False, False)
    root.mainloop()


MainGUI()
