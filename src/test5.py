import cleantext

question = "Who is the main calligrapher?"
context1 = """
8.37
Ranked #206Popularity #313Members 670,618Summer 2014TVKinema Citrus
Add to My List
Select
Episodes: 
0
/12
CM 15-second version
play
More videos
Edit
Synopsis
-Advertisement-
Seishuu Handa is an up-and-coming calligrapher: young, handsome, talented, and unfortunately, a narcissist to boot. When a veteran labels his award-winning piece as "unoriginal," Seishuu quickly loses his cool with severe repercussions.

As punishment, and also in order to aid him in self-reflection, Seishuu's father exiles him to the Goto Islands, far from the comfortable Tokyo lifestyle the temperamental artist is used to. Now thrown into a rural setting, Seishuu must attempt to find new inspiration and develop his own unique art style—that is, if boisterous children (headed by the frisky Naru Kotoishi), fujoshi middle schoolers, and energetic old men stop barging into his house! The newest addition to the intimate and quirky Goto community only wants to get some work done, but the islands are far from the peaceful countryside he signed up for. Thanks to his wacky neighbors who are entirely incapable of minding their own business, the arrogant calligrapher learns so much more than he ever hoped to.

[Written by MAL Rewrite]
"""
context2 = """
8.46
Ranked #152Popularity #6Members 3,091,090Spring 2019TVufotable
Add to My List
Select
Episodes: 
0
/26
PV 3 Aniplex USA version
play
More videos
Edit
Synopsis
Ever since the death of his father, the burden of supporting the family has fallen upon Tanjirou Kamado's shoulders. Though living impoverished on a remote mountain, the Kamado family are able to enjoy a relatively peaceful and happy life. One day, Tanjirou decides to go down to the local village to make a little money selling charcoal. On his way back, night falls, forcing Tanjirou to take shelter in the house of a strange man, who warns him of the existence of flesh-eating demons that lurk in the woods at night.

When he finally arrives back home the next day, he is met with a horrifying sight—his whole family has been slaughtered. Worse still, the sole survivor is his sister Nezuko, who has been turned into a bloodthirsty demon. Consumed by rage and hatred, Tanjirou swears to avenge his family and stay by his only remaining sibling. Alongside the mysterious group calling themselves the Demon Slayer Corps, Tanjirou will do whatever it takes to slay the demons and protect the remnants of his beloved sister's humanity.

[Written by MAL Rewrite]
"""

print(question)
clean_question = cleantext.clean(question, extra_spaces=True, lowercase=True, numbers=True, punct=True)
print("---CLEANED---")
print(clean_question)
print("-"*8)
print(context1)
print("---CLEANED---")
clean_context1 = cleantext.clean(context1, extra_spaces=True, lowercase=True, numbers=True, punct=True)
print(clean_context1)
print("-"*8)
print(context2)
print("---CLEANED---")
clean_context2 = cleantext.clean(context2, extra_spaces=True, lowercase=True, numbers=True, punct=True)
print(clean_context2)
print("-"*8)
