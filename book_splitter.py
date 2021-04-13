import re

# Add author and name of books to dictionary as shown below.
# Note that books should not have the ".txt" suffix-ed.

#authors = {"charles_dickens": ["greatex", "olivert", "twocities"] }
#authors = {"fyodor_dostoevsky": ["crimep", "idiot", "possessed"]} 
#authors = {"mark_twain": ["toms", "huckfinn", "connecticutyankee", "princepauper"]}
authors = {"mark_twain": ["puddnheadWilson"]}

# Specify text chunk length
text_length = 50000

# Specify the folder name which contains the books to be split
folderName = "novels/"

for author in authors:
    for bookName in authors[author]:
        print("Splitting book now: {}".format(bookName))
        bookFilepath = folderName + author + "/" + bookName + ".txt"
        # print(bookFilepath)

        # read text in book
        with open(bookFilepath, encoding='utf-8', errors='ignore') as f:
            text = f.read()[1:]
        text = text.replace("\n", " ")
        text = text.split(" ")

        # concatenate till 50k words then write
        # need to check length of remaining array see if < 50k. If less than 50k,
        counter = 1
        end_of_sentence_regex = r'(?<!Dr|Mr|Ms|Jr|Sr|St)(?<!Mrs|Rev)[\.!\?]"?'
        while len(text) > text_length:
            remaining_text = text[text_length:]
            to_append = []
            # end split text with sentence
            if len(re.findall(end_of_sentence_regex, text[text_length - 1])) <= 0:
                # append remaining words of sentence to split text
                while len(re.findall(end_of_sentence_regex, remaining_text[0])) <= 0:
                    to_append.append(remaining_text[0])
                    remaining_text = remaining_text[1:]
                to_append.append(remaining_text[0])
                remaining_text = remaining_text[1:]

            if len(remaining_text) < text_length:
                text_to_write = " ".join(text)
                text = []
            else:
                text_to_write = text[0:text_length]
                text_to_write.extend(to_append)
                text_to_write = " ".join(text_to_write)
                text = remaining_text
            writePath = "split_novels/" + author + "/" + bookName + str(counter) + ".txt"
            f = open(writePath, "x")
            f.write(text_to_write)
            f.close()
            print("Done writing file {} of book: {}".format(counter, bookName))
            counter += 1
    
print("Book split complete!")
    