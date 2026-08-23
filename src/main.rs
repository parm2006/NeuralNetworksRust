//following rust tutorial for now before NN

fn ownership() {
    //mem management w heap in rust
    //everythin is immutable by default, it stops race conditions between threads, unless stateed to be mutable (mut)

    //ownership in Rust
    /*
    ownership is a set of rules for rust to managem memor. -> Set of rules for the compiler to check to make sure that no errors will be possible at compile time in the code.
    stops dangling pts errors etc like in c++.

    ownership makes program faster, but makes compilation slower. compliation is slower for rust but runtime vry fast,
    ownerhsip is there for things on the heap mostly not stack

    data must always have one boyfriend. boyfriends can be changed and passed on. if no bf, then i will die.

    stack variables on the stack dont really matter.
    they are owned by the calling function that they are in. that function that they are made in is the owner.
    the inside variables go out of scope when the calling function stack frame is popped off the stack


    heap:
    anytime something is created on the heap, we need a pointer on the stack to it. That is the owner, (how does work. what defines a variable, like a matrix has one owner or nxn owner,s what if i put a object inside of another)
    also rust stopts double owners, if i have int* s1 = new 4, s2 = s1. both s1 and s2 would point to 4, but now the owner of 4 changes, the variable moves on and the old owner becomes invalid, so the object/variable can only retain
    a single owner at all times. if we have multiple, the olderones gets ignored and (killed i think so) and only the last one is considered the owner, so when it goes away, 4 will too.
    you can not have two boyfriends.  avoiding double owners in heap = no double free error

     */
    let s1: String = String::from("hello");
    println!("{}", s1);
    //this is fine because we havent moved
    let s2: String = s1;
    //println!("{}",s1); //error since s1 is no longer the owner and so we cant access it. The value is being borrowed after it was already moved
    println!("{}", s2);
    //rust says that whenever the current owner of a data object dies, then the heap automatically dies, so we dont need manual Mem Manage or garbage collector, instead the owners pet automatically dies anytime an owner dies

    //values can also be moved or borrowed by going into a function (cheating on their owner) when you pass a variable it gets copied fully cuz its only on the stack, but for heap variables, we create a new stack pointer to it, but its the same heaping girlfriend variable

    //if you pass in a string, you copy a pointer to its loc on the heap as the new guy, but its still the same char[].

    //if you use object.clone(), then we allocated a new girlfriend instead of the same on, so we are chill, the twin has a new bf, but the og bf stays.
    //you canreturn owner ship via returning something back to the caller function and then it my_string will take back ownership from stealerstring which was passed into a function, returning doesnt bounce back but instead just tells the compiler that this is the new owner.
}

fn loops() {
    //loops and conditionals
    let is_even = true;

    if is_even {
        println!("This is even!");
    } else if !is_even {
        println!("This is not even");
    } else {
        println!()
    }

    // for i in 0..100{
    //     print!("{}",i);
    // }

    let sentence: String = String::from("my name is parth");
    let firstword = get_first_word(sentence);
    println!("The first word is: {}", firstword);

    let n = 50;
    for _i in 0..n {
        println!("Hello dude");
    }
}

fn get_first_word(sentence: String) -> String {
    //you must define the return type of the function whenever you create it.
    let mut ans: String = String::from("");
    for char in sentence.chars() {
        if char == ' ' {
            break;
        }
        ans.push_str(char.to_string().as_str());
    }
    return ans;
}

fn strings() {
    let x: &str = "strisngs";

    //not a fixed type cuz there are not fixed length like vector so its hard to keep it
    //since it can change at runtime,

    //string mutation is hard for rust because you have to change the allocated space, not automatica, unlike nums which have fixed space
    //instead use string::from("...")
    let mut words: String = String::from("hello ");

    let greet = String::from("hello world");

    println!("{}", greet);

    let char1: Option<char> = greet.chars().nth(0); //so the type is optionally a character, or it will be none, so rust wont let us directly print this char, we have to use match/switch for the case if it is and hte case if its not.

    match char1 {
        //pattern matching
        Some(c) => print!("{}", c),
        None => print!("No char at this location"),
    }

    print!("char1:{}", char1.unwrap()) //unwrap will unwrap the option type and so we will be chill. it will cause a runtime exception
}

fn bool() {
    let mut is_male = false;

    //let isnotmale = !is_male;

    if is_male {
        print!("You are male");
    } else {
        is_male = true;
    }
    if is_male {
        print!("Now you are male")
    }
}

fn vars() {
    //simple variables in rust
    //define vars with let
    let x = 5;
    //let is the init-er
    let w: i32 = 6; //we can also declare the type of the variable we want to make it. //these are immutable by default

    //    let z:u8 = -4; //error because unsigned cant be negative
    let y: u32 = 1000;

    let f: f32 = 1.001; // float

    print!("x :{}", f);

    //runtime overflow: cant be checked by compiler obviously
}
