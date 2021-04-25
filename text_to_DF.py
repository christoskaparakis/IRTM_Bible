import urllib.request
from urllib.request import urlopen
import pandas as pd

url = 'https://raw.githubusercontent.com/ErikSchierboom/sentencegenerator/master/samples/the-king-james-bible.txt'

file = urlopen(url)


df = pd.DataFrame(columns=["book", "chapter", "verse", "text"])
books = pd.DataFrame(columns=["id", "book"])

count = 0
gap = 0
for line in file:
    count += 1
    # print(line)
    decoded_line = line.decode("utf-8").rstrip("\n")
    # print(decoded_line)

    if count == 1:
        bookid = 1
        bookname = decoded_line

        books = books.append({
                            "id": bookid,
                            "book": bookname
                        }, ignore_index=True)
        continue

    if gap == 4:
        bookid += 1
        bookname = decoded_line

        books = books.append({
            "id": bookid,
            "book": bookname
        }, ignore_index=True)
        pass

    if decoded_line == "":
        gap += 1
        continue
    else:
        gap = 0



    if str.isdigit(decoded_line[0]):
        chapter = decoded_line[0]
        verse = decoded_line[2]
        text = decoded_line[4:]
        try:
            if next(file).decode("utf-8").rstrip("\n") == "":
                df = df.append({
                    "book": bookid,
                    "chapter": chapter,
                    "verse": verse,
                    "text": text
                }, ignore_index=True)
                print(bookid, chapter, verse, text)
        except:
            print("An exception occurred")

    else:
        text = text + " " + decoded_line
        try:
            if next(file).decode("utf-8").rstrip("\n") == "":
                df = df.append({
                    "book": bookid,
                    "chapter": chapter,
                    "verse": verse,
                    "text": text
                }, ignore_index=True)
                print(bookid, chapter, verse, text)
        except:
            print("An exception occurred")

