import chardet
import os

def readFile(inputPath):
    try:
        if not os.path.exists(inputPath):
            print(f"错误: 文件不存在: {inputPath}")
            return "", ""
            
        # 检查文件大小
        file_size = os.path.getsize(inputPath)
        if file_size == 0:
            print(f"错误: 文件为空: {inputPath}")
            return "", ""
            
        f = open(inputPath, 'rb')
        lines = f.readlines()
        
        if not lines:
            print(f"错误: 文件没有内容: {inputPath}")
            f.close()
            return "", ""
            
        # 处理标题
        line = lines[0]
        line = line.strip()
        if not line:
            title = ""
        else:
            try:
                f_charInfo = chardet.detect(line)
                encoding = f_charInfo.get('encoding', 'utf-8')
                title = line.decode(encoding)
            except Exception as e:
                print(f"解码标题时出错: {str(e)}")
                title = ""

        # 处理正文
        body = ""
        for line in lines[1:]:
            line = line.strip()
            if line == b'':
                continue
            try:
                f_charInfo = chardet.detect(line)
                encoding = f_charInfo.get('encoding', 'utf-8')
                f_read_decode = line.decode(encoding)
                if f_read_decode != "":
                    body += f_read_decode
            except Exception as e:
                print(f"解码正文行时出错: {str(e)}")
                continue

        f.close()
        return title, body
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return "", ""
